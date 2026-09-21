"""Audio primitives shared by training, the ASR server and the edge firmware.

This module is the single source of truth for every signal-processing step in
the project.  It is deliberately dependency-light (numpy + stdlib ``wave``) and
bit-reproducible: the C implementation that runs on the microcontroller
(``edge/common/kws_frontend.c``) mirrors these functions one-for-one, and
``tests/test_frontend_parity.py`` proves the two agree sample for sample.

Reference: SIH26172 - "Low Latency and Efficient Voice Activator for Edge
Devices" (ISRO).
"""

from __future__ import annotations

import hashlib
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Canonical audio format for the whole project
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
"""Sample rate of the acoustic front-end (Hz).  16 kHz is the sweet spot for
keyword spotting: it covers the 0-8 kHz band that carries all speech
intelligibility, while keeping the mel filterbank small enough to fit in the
microcontroller's SRAM."""

FRAME_LENGTH = 480  # 30 ms
FRAME_HOP = 320  # 20 ms


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------
# Deliberately a hand-written RIFF parser rather than the stdlib ``wave`` module
# (which cannot read WAVE_FORMAT_EXTENSIBLE before Python 3.12, and which rejects
# 24-bit and float WAVs).  The corpora we use are a real-world mix: 16-bit
# 16 kHz mono in WAVE_FORMAT_PCM, plus 24-bit and 16-bit 44.1/48 kHz stereo in
# WAVE_FORMAT_EXTENSIBLE.  Getting this wrong silently dropped 88% of the GSC
# negatives and a third of the hard negatives on the first attempt, so it gets
# an explicit implementation and its own tests.
WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_ALAW = 0x0006
WAVE_FORMAT_MULAW = 0x0007
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_INT24_SCALE = 8388608.0  # 2**23
_INT32_SCALE = 2147483648.0  # 2**31


def parse_wav_header(path: str | Path, data: bytes | None = None) -> dict:
    """Parse the RIFF header and return a description of the audio payload."""
    if data is None:
        data = Path(path).read_bytes()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")

    fmt: dict | None = None
    payload = bytearray()
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body = data[pos + 8 : pos + 8 + size]
        if chunk_id == b"fmt ":
            if len(body) < 16:
                raise ValueError(f"{path}: truncated fmt chunk")
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack_from(
                "<HHIIHH", body, 0
            )
            valid_bits = bits
            if audio_format == WAVE_FORMAT_EXTENSIBLE:
                if len(body) < 26:
                    raise ValueError(f"{path}: truncated extensible fmt chunk")
                (valid_bits,) = struct.unpack_from("<H", body, 18)
                (audio_format,) = struct.unpack_from("<H", body, 24)  # first word of the sub-format GUID
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "sample_rate": sample_rate,
                "bits": bits,
                "valid_bits": valid_bits,
            }
        elif chunk_id == b"data":
            payload.extend(body)
        pos += 8 + size + (size & 1)

    if fmt is None:
        raise ValueError(f"{path}: no fmt chunk")
    if not payload:
        raise ValueError(f"{path}: no data chunk")
    fmt["n_bytes"] = len(payload)
    fmt["n_frames"] = len(payload) // max(1, fmt["channels"] * (fmt["bits"] // 8))
    return fmt


def _decode_pcm(raw: bytes, fmt: dict, path: str | Path) -> np.ndarray:
    audio_format, bits = fmt["audio_format"], fmt["bits"]
    if audio_format == WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            return np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if bits == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float32)
        raise ValueError(f"{path}: unsupported float width {bits}")
    if audio_format == WAVE_FORMAT_PCM:
        if bits == 8:
            return ((np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128) / 128.0).astype(np.float32)
        if bits == 16:
            return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).astype(np.float32)
        if bits == 24:
            b = np.frombuffer(raw, dtype=np.uint8)
            b = b[: (b.size // 3) * 3].reshape(-1, 3).astype(np.int32)
            ints = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
            ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
            return (ints.astype(np.float32) / _INT24_SCALE).astype(np.float32)
        if bits == 32:
            return (np.frombuffer(raw, dtype="<i4").astype(np.float32) / _INT32_SCALE).astype(np.float32)
    raise ValueError(f"{path}: unsupported WAV format {audio_format} / {bits} bit")


def read_wav_bytes(data: bytes, name: str = "<bytes>") -> tuple[np.ndarray, int]:
    """Decode WAV bytes into ``float32`` in [-1, 1).

    Handles WAVE_FORMAT_PCM (8/16/24/32-bit), IEEE float (32/64-bit) and
    WAVE_FORMAT_EXTENSIBLE, with any channel count.  Returns
    ``(samples, sample_rate)`` where samples has shape ``(n,)`` or
    ``(n, channels)``.  This is the entry point used by the server for uploads.
    """
    fmt = parse_wav_header(name, data)
    pos, payload = 12, bytearray()
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        if chunk_id == b"data":
            payload.extend(data[pos + 8 : pos + 8 + size])
        pos += 8 + size + (size & 1)

    x = _decode_pcm(bytes(payload), fmt, name)
    channels = fmt["channels"]
    if channels > 1:
        n = (x.size // channels) * channels
        x = x[:n].reshape(-1, channels)
    return x, int(fmt["sample_rate"])


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV file into ``float32`` in the range [-1, 1)."""
    return read_wav_bytes(Path(path).read_bytes(), str(path))


def wav_duration(path: str | Path) -> float:
    """Duration in seconds, read from the header only (no payload decode)."""
    fmt = parse_wav_header(path)
    return float(fmt["n_frames"]) / max(1.0, float(fmt["sample_rate"]))


def write_wav(path: str | Path, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono ``float32`` audio as a 16-bit PCM WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    pcm = np.clip(x, -1.0, 1.0 - 2**-15)
    pcm = np.round(pcm * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def to_mono(x: np.ndarray) -> np.ndarray:
    """Downmix to a single channel (mean of channels)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.reshape(-1)


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """Band-limited polyphase resampling (scipy under the hood).

    The corpora used here are a mix of 16 kHz / 44.1 kHz / 48 kHz, so this is
    the first normalisation step of the data pipeline.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if sr_in == sr_out:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read -> downmix -> resample to the canonical 16 kHz mono format."""
    x, sr = read_wav(path)
    return resample(to_mono(x), sr, target_sr)


# ---------------------------------------------------------------------------
# Level helpers
# ---------------------------------------------------------------------------
_EPS = 1e-12


def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + _EPS))


def db(x: np.ndarray | float) -> float:
    """Full-scale dBFS level of a signal or of an RMS value."""
    value = x if np.isscalar(x) else rms(np.asarray(x))
    return float(20.0 * np.log10(max(float(value), _EPS)))


def peak_normalize(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Scale so that the loudest sample sits at ``peak``."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(x))) if x.size else 0.0
    return x * (peak / m) if m > _EPS else x.copy()


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply a constant gain in dB, clipping to [-1, 1)."""
    return np.clip(np.asarray(x, dtype=np.float32) * (10.0 ** (gain_db / 20.0)), -1.0, 1.0 - 2**-15)


# ---------------------------------------------------------------------------
# G.711 mu-law - the wire format between the edge device and the ASR server
# ---------------------------------------------------------------------------
ULAW_BIAS = 0x84  # 132
ULAW_CLIP = 32635


def ulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """Encode int16 PCM to 8-bit G.711 mu-law (bit-exact, vectorised).

    mu-law is what makes the "minimal data overhead" requirement concrete:
    64 kbit/s at 8 kHz instead of the 256 kbit/s of raw 16-bit/16 kHz audio - a
    4x reduction - at a speech quality that ASR back-ends are trained on.

    The exponent is the position of the highest set bit of ``(|x| + bias)`` above
    bit 7: ``e = clamp(floor(log2(s)) - 7, 0, 7)``.  Computing it with a descending
    threshold sweep is exact and branch-free.  (An earlier version folded the
    loop into ``8 - shift`` and produced every code off by a factor of two - the
    encoder/decoder round-trip test in ``tests/test_audio.py`` catches exactly
    that, which is why that test compares against a literal transcription of the
    ITU-T reference rather than against another vectorised implementation.)
    """
    x = np.asarray(pcm).astype(np.int32).reshape(-1)
    sign = np.where(x < 0, 0x80, 0x00)
    mag = np.minimum(np.abs(x), ULAW_CLIP) + ULAW_BIAS
    # ascending sweep: the largest satisfied threshold must win the last write
    exponent = np.zeros_like(mag)
    for e in range(1, 8):
        exponent = np.where(mag >= (1 << (e + 7)), e, exponent)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return ((~(sign | (exponent << 4) | mantissa)) & 0xFF).astype(np.uint8)


def ulaw_decode(ulaw: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 mu-law back to int16 PCM."""
    u = np.asarray(ulaw).astype(np.int32).reshape(-1)
    u = ~u & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + ULAW_BIAS) << exponent
    sample = sample - ULAW_BIAS
    return np.where(sign != 0, -sample, sample).astype(np.int16)


def pcm16_from_float(x: np.ndarray) -> np.ndarray:
    """float32 [-1,1) -> int16 PCM with round-half-away-from-zero."""
    return np.round(np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0 - 2**-15) * 32768.0).astype(np.int16)


def float_from_pcm16(x: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 [-1,1)."""
    return np.asarray(x).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def n_frames(n_samples: int, frame_length: int = FRAME_LENGTH, hop: int = FRAME_HOP) -> int:
    """Number of *whole* frames of length ``frame_length`` that fit with ``hop``.

    Mirrors the streaming front-end: the MCU never zero-pads, it simply keeps
    feeding the ring buffer and emits a frame once enough samples exist.
    """
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop


def frame_signal(
    x: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop: int = FRAME_HOP,
    pad_end: bool = False,
) -> np.ndarray:
    """Split a 1-D signal into overlapping frames, shape ``(n_frames, frame_length)``."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if pad_end:
        needed = frame_length - max(len(x), 1)
        if needed > 0:
            x = np.pad(x, (0, needed))
        n = n_frames(x.size, frame_length, hop)
        total = n * hop + (frame_length - hop)
        if total > x.size:
            x = np.pad(x, (0, total - x.size))
    n = n_frames(x.size, frame_length, hop)
    if n <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    idx = np.arange(frame_length)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def fixed_window(x: np.ndarray, n_samples: int, offset: int = 0, pad_value: float = 0.0) -> np.ndarray:
    """Deterministically place ``x`` inside an ``n_samples`` window at ``offset``.

    Used for training (random offsets) *and* for the streaming simulator, so
    the two cannot drift apart.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.full(int(n_samples), float(pad_value), dtype=np.float32)
    if x.size == 0:
        return out
    src_start = max(0, -offset)
    dst_start = max(0, offset)
    count = min(x.size - src_start, n_samples - dst_start)
    if count > 0:
        out[dst_start : dst_start + count] = x[src_start : src_start + count]
    return out


def random_window(
    x: np.ndarray, n_samples: int, rng: np.random.Generator, keep_inside: bool = True
) -> np.ndarray:
    """Place a clip at a random offset inside a fixed-size analysis window.

    ``keep_inside=True``  -> the whole utterance is inside the window (the
    model must learn position invariance).
    ``keep_inside=False`` -> the utterance may be clipped by the window edge,
    which teaches the detector to stay quiet while a keyword is only *partly*
    visible.  This is what kills the classic "fires twice" failure mode.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(int(n_samples), dtype=np.float32)
    if keep_inside and x.size <= n_samples:
        offset = int(rng.integers(0, n_samples - x.size + 1))
    else:
        offset = int(rng.integers(-x.size + 1, n_samples))
    return fixed_window(x, n_samples, offset)


# ---------------------------------------------------------------------------
# Synthetic noise bank
# ---------------------------------------------------------------------------
def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise generated in the frequency domain."""
    w = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.maximum(np.arange(1, w.size + 1), 1.0)
    spec = w / np.sqrt(f)
    spec[0] = 0.0
    return np.fft.irfft(spec, n).astype(np.float32)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f^2 noise (air-conditioner / fan rumble)."""
    w = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.maximum(np.arange(1, w.size + 1), 1.0)
    spec = w / f
    spec[0] = 0.0
    return np.fft.irfft(spec, n).astype(np.float32)


def hum_noise(n: int, sr: int = SAMPLE_RATE, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mains hum at 50 Hz + harmonics (very common in Indian field recordings)."""
    rng = rng or np.random.default_rng(0)
    t = np.arange(n, dtype=np.float32) / sr
    phase = float(rng.uniform(0, 2 * np.pi))
    out = np.zeros(n, dtype=np.float32)
    for k, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
        out += amp * np.sin(2 * np.pi * 50.0 * k * t + phase)
    return out / 2.0


_NOISE_BANK = {
    "white": white_noise,
    "pink": pink_noise,
    "brown": brown_noise,
}


def synth_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample one of the synthetic background-noise generators."""
    if kind == "hum":
        return hum_noise(n, rng=rng)
    return _NOISE_BANK[kind](n, rng)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def clip_id(path: str | Path) -> str:
    """Stable short identifier for an audio file (used in manifests)."""
    return hashlib.sha1(Path(path).name.encode()).hexdigest()[:12]


def speaker_id_from_gsc(filename: str) -> str:
    """Google Speech Commands filenames are ``<speaker>_nohash_<n>.wav``.

    The speaker hash lets us build *speaker-disjoint* splits, which is the only
    honest way to report keyword-spotting accuracy (random splits leak the same
    speaker into train and test and inflate the numbers).
    """
    return Path(filename).stem.split("_nohash_")[0]


def iter_wavs(root: str | Path) -> Iterable[Path]:
    """Yield every ``.wav`` below ``root`` in a deterministic order."""
    yield from sorted(Path(root).rglob("*.wav"))


@dataclass(frozen=True)
class Clip:
    """One item of the training corpus (before feature extraction)."""

    path: Path
    label: int  # 1 = keyword, 0 = not-keyword
    speaker: str
    source: str
    split: str
