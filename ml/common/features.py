"""Log-mel acoustic front-end + fixed-point quantisation.

Design constraints that shaped this file
----------------------------------------
The same front-end has to run in three places and produce *identical* numbers:

1. training / evaluation (numpy, this file),
2. the ASR server's HTTP/WS ingest path (numpy),
3. the ESP32 firmware (C, ``edge/common/kws_frontend.c``).

So it is written as a plain, branch-free pipeline with explicitly stored
constant tables (mel matrix, window, quantisation scale/zero-point).  The mel
matrix and the int8 quantisation parameters are exported to C headers by
``edge/tools/export_frontend_header.py`` and the parity is checked in
``tests/test_frontend_parity.py``.

Feature spec
------------
* 16 kHz mono, 30 ms Hann window, 20 ms hop (480/320 samples)
* 512-point real FFT
* 40 HTK-mel triangular filters between 20 Hz and 7600 Hz
* natural log of the mel energies, floored at ``log_floor``
* affine quantisation to int8 with a *fixed* scale/zero-point that is also
  baked into the TFLite model's input tensor, so the MCU never needs a
  floating-point quantiser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import FRAME_HOP, FRAME_LENGTH, SAMPLE_RATE, frame_signal

# ---------------------------------------------------------------------------
# Mel filterbank
# ---------------------------------------------------------------------------
MEL_LOWER_HZ = 20.0
MEL_UPPER_HZ = 7600.0
NUM_MEL_BINS = 40
FFT_SIZE = 512
CONTEXT_FRAMES = 49  # ~0.98 s of context, the streaming detection window
LOG_FLOOR = 1e-6


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """HTK mel scale (the one used by Kaldi / the classic Speech Commands papers)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def hann_periodic(n: int) -> np.ndarray:
    """Periodic Hann window, matching ``tf.signal.hann_window(periodic=True)``.

    Periodic (not symmetric) is the correct choice for spectral analysis and is
    what the C implementation uses, so it must be spelled out explicitly.
    """
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    fft_size: int = FFT_SIZE,
    num_bins: int = NUM_MEL_BINS,
    lower_hz: float = MEL_LOWER_HZ,
    upper_hz: float = MEL_UPPER_HZ,
) -> np.ndarray:
    """Triangular mel filterbank, shape ``(num_bins, fft_size // 2 + 1)``.

    Area-normalised (Slaney style) so that a flat spectrum yields a flat mel
    output; this keeps the int8 dynamic range of the features sane.
    """
    n_freqs = fft_size // 2 + 1
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    mel_lo, mel_hi = hz_to_mel(lower_hz), hz_to_mel(upper_hz)
    mel_points = np.linspace(mel_lo, mel_hi, num_bins + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((num_bins, n_freqs), dtype=np.float32)
    for i in range(num_bins):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (freqs - left) / max(center - left, 1e-9)
        falling = (right - freqs) / max(right - center, 1e-9)
        tri = np.clip(np.minimum(rising, falling), 0.0, None)
        # Slaney-style area normalisation: divide by the filter bandwidth so
        # wide (high-frequency) filters do not dominate the log-mel output.
        fb[i] = (tri * 2.0 / max(right - left, 1e-9)).astype(np.float32)
    return fb


# ---------------------------------------------------------------------------
# Fixed-point parameters
# ---------------------------------------------------------------------------
@dataclass
class QuantParams:
    """Affine int8 quantisation: ``q = clip(round(x / scale) + zero_point)``."""

    scale: float
    zero_point: int

    def quantize(self, x: np.ndarray) -> np.ndarray:
        q = np.round(np.asarray(x, dtype=np.float64) / self.scale) + self.zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float32) - self.zero_point) * self.scale

    def to_dict(self) -> dict:
        return {"scale": float(self.scale), "zero_point": int(self.zero_point), "dtype": "int8"}

    @staticmethod
    def from_dict(d: dict) -> "QuantParams":
        return QuantParams(float(d["scale"]), int(d["zero_point"]))


# ---------------------------------------------------------------------------
# Front-end
# ---------------------------------------------------------------------------
@dataclass
class FrontendParams:
    """Everything needed to turn a waveform into the model's input tensor."""

    sample_rate: int = SAMPLE_RATE
    frame_length: int = FRAME_LENGTH
    frame_hop: int = FRAME_HOP
    fft_size: int = FFT_SIZE
    num_mel_bins: int = NUM_MEL_BINS
    lower_hz: float = MEL_LOWER_HZ
    upper_hz: float = MEL_UPPER_HZ
    context_frames: int = CONTEXT_FRAMES
    log_floor: float = LOG_FLOOR
    preemphasis: float = 0.0
    quant: QuantParams | None = None  # set after calibration/export
    _window: np.ndarray | None = field(default=None, repr=False, compare=False)
    _mel: np.ndarray | None = field(default=None, repr=False, compare=False)

    # -- derived resources, cached lazily ---------------------------------
    @property
    def window(self) -> np.ndarray:
        if self._window is None:
            self._window = hann_periodic(self.frame_length)
        return self._window

    @property
    def mel_matrix(self) -> np.ndarray:
        if self._mel is None:
            self._mel = mel_filterbank(
                self.sample_rate, self.fft_size, self.num_mel_bins, self.lower_hz, self.upper_hz
            )
        return self._mel

    @property
    def window_samples(self) -> int:
        """Samples spanned by one analysis window (= hop*(context-1)+frame_length)."""
        return self.frame_hop * (self.context_frames - 1) + self.frame_length

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "frame_length": self.frame_length,
            "frame_hop": self.frame_hop,
            "frame_ms": self.frame_length / self.sample_rate * 1000.0,
            "hop_ms": self.frame_hop / self.sample_rate * 1000.0,
            "fft_size": self.fft_size,
            "num_mel_bins": self.num_mel_bins,
            "lower_hz": self.lower_hz,
            "upper_hz": self.upper_hz,
            "context_frames": self.context_frames,
            "log_floor": self.log_floor,
            "preemphasis": self.preemphasis,
            "window": "hann_periodic",
            "quant": self.quant.to_dict() if self.quant else None,
        }

    @staticmethod
    def from_dict(d: dict) -> "FrontendParams":
        return FrontendParams(
            sample_rate=int(d.get("sample_rate", SAMPLE_RATE)),
            frame_length=int(d.get("frame_length", FRAME_LENGTH)),
            frame_hop=int(d.get("frame_hop", FRAME_HOP)),
            fft_size=int(d.get("fft_size", FFT_SIZE)),
            num_mel_bins=int(d.get("num_mel_bins", NUM_MEL_BINS)),
            lower_hz=float(d.get("lower_hz", MEL_LOWER_HZ)),
            upper_hz=float(d.get("upper_hz", MEL_UPPER_HZ)),
            context_frames=int(d.get("context_frames", CONTEXT_FRAMES)),
            log_floor=float(d.get("log_floor", LOG_FLOOR)),
            preemphasis=float(d.get("preemphasis", 0.0)),
            quant=QuantParams.from_dict(d["quant"]) if d.get("quant") else None,
        )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def log_mel_spectrogram(
    x: np.ndarray,
    params: FrontendParams | None = None,
    frame_count: int | None = None,
) -> np.ndarray:
    """Waveform -> log-mel matrix of shape ``(n_frames, num_mel_bins)``.

    When ``frame_count`` is given the result is trimmed/padded to exactly that
    many frames, which is how the streaming path and the training path are kept
    in agreement.
    """
    params = params or FrontendParams()
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if params.preemphasis > 0.0 and x.size > 1:
        x = np.concatenate([x[:1], x[1:] - params.preemphasis * x[:-1]]).astype(np.float32)

    frames = frame_signal(x, params.frame_length, params.frame_hop)  # (T, 480)
    if frames.shape[0] == 0:
        mel = np.full((0, params.num_mel_bins), params.log_floor, dtype=np.float32)
    else:
        # Order matters: the Hann window belongs to the *frame*, so it is applied
        # to the 480 real samples before the frame is zero-padded to the 512-point
        # FFT.  (Padding first and then multiplying by a 480-sample window is a
        # shape error - and if the window were padded too, it would silently
        # window the zeros and change every feature.)
        windowed = frames[:, : params.frame_length] * params.window[None, :]
        if params.frame_length < params.fft_size:
            windowed = np.pad(
                windowed, ((0, 0), (0, params.fft_size - params.frame_length)), mode="constant"
            )
        spec = np.fft.rfft(windowed.astype(np.float64), n=params.fft_size, axis=1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        mel = power @ params.mel_matrix.T
        mel = np.log(np.maximum(mel, params.log_floor)).astype(np.float32)

    return _fix_frames(mel, frame_count, params)


def _fix_frames(mel: np.ndarray, frame_count: int | None, params: FrontendParams) -> np.ndarray:
    if frame_count is None:
        return mel
    n = int(frame_count)
    if mel.shape[0] == n:
        return mel
    if mel.shape[0] > n:
        return mel[:n]
    pad = np.full((n - mel.shape[0], params.num_mel_bins), np.log(params.log_floor), dtype=np.float32)
    return np.concatenate([pad, mel], axis=0)  # left-pad = "no audio yet"


def features_from_waveform(
    x: np.ndarray,
    params: FrontendParams | None = None,
    quantize: bool = False,
) -> np.ndarray:
    """Waveform -> fixed-size feature tensor for one model invocation.

    Shape ``(context_frames, num_mel_bins)`` in float32, or int8 when
    ``quantize=True`` and the params carry quantisation constants.
    """
    params = params or FrontendParams()
    mel = log_mel_spectrogram(x, params, frame_count=params.context_frames)
    if not quantize:
        return mel.astype(np.float32)
    if params.quant is None:
        raise ValueError("quantize=True requires FrontendParams.quant to be set")
    return params.quant.quantize(mel)


def features_from_file(
    path: str | Path,
    params: FrontendParams | None = None,
    quantize: bool = False,
    offset: int = 0,
) -> np.ndarray:
    """Convenience wrapper: load a WAV and extract one analysis window."""
    from .audio import load_audio

    params = params or FrontendParams()
    x = load_audio(path, params.sample_rate)
    window = params.window_samples
    if x.size < window or offset:
        from .audio import fixed_window

        x = fixed_window(x, window, offset=offset)
    else:
        x = x[-window:]
    return features_from_waveform(x, params, quantize=quantize)


# ---------------------------------------------------------------------------
# Batch feature extraction (training)
# ---------------------------------------------------------------------------
def log_mel_batch(
    waveforms: np.ndarray,
    params: FrontendParams | None = None,
    frame_count: int | None = None,
) -> np.ndarray:
    """Vectorised log-mel for a batch of equal-length waveforms ``(B, N)``."""
    params = params or FrontendParams()
    x = np.asarray(waveforms, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("expected a (batch, samples) array")
    if params.preemphasis > 0.0:
        x = np.concatenate([x[:, :1], x[:, 1:] - params.preemphasis * x[:, :-1]], axis=1)

    b, n = x.shape
    n_f = 1 + (n - params.frame_length) // params.frame_hop if n >= params.frame_length else 0
    if n_f <= 0:
        raise ValueError(f"waveform too short for one frame: {n} samples")
    idx = np.arange(params.frame_length)[None, :] + params.frame_hop * np.arange(n_f)[:, None]
    frames = x[:, idx]  # (B, T, 480)
    frames = frames * params.window[None, None, :]
    spec = np.fft.rfft(frames.astype(np.float64), n=params.fft_size, axis=2)
    power = (spec.real**2 + spec.imag**2).astype(np.float32)
    mel = power @ params.mel_matrix.T  # (B, T, 40)
    mel = np.log(np.maximum(mel, params.log_floor)).astype(np.float32)
    return _fix_frames_batch(mel, frame_count, params)


def _fix_frames_batch(mel: np.ndarray, frame_count: int | None, params: FrontendParams) -> np.ndarray:
    if frame_count is None:
        return mel
    n = int(frame_count)
    t = mel.shape[1]
    if t == n:
        return mel
    if t > n:
        return mel[:, :n]
    pad = np.full((mel.shape[0], n - t, params.num_mel_bins), np.log(params.log_floor), dtype=np.float32)
    return np.concatenate([pad, mel], axis=1)


def calibrate_quant_params(features: np.ndarray, tight: bool = True) -> QuantParams:
    """Derive the int8 affine mapping from a representative feature sample.

    ``tight`` clips the observed range at the 0.1/99.9 percentiles: the tail of
    the log-mel distribution is dominated by near-silent frames and spending
    int8 levels on them costs real accuracy.
    """
    f = np.asarray(features, dtype=np.float32)
    if tight:
        lo, hi = np.percentile(f, [0.1, 99.9])
    else:
        lo, hi = float(f.min()), float(f.max())
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    scale = (hi - lo) / 255.0
    zero_point = int(round(-128.0 - lo / scale))
    zero_point = max(-128, min(127, zero_point))
    return QuantParams(scale=scale, zero_point=zero_point)
