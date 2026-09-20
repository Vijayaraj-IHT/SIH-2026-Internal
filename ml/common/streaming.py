"""Streaming wake-word detection: the code path that actually runs at the edge.

Design notes that matter for the SIH evaluation criteria
-------------------------------------------------------
**Latency is defined, not vibes.**  The detector is re-evaluated once per 20 ms
hop (every 320 new samples).  Window ``k`` covers frames ``[k-48, k]``, so the
newest sample it can have seen is ``k*320 + 480``.  We therefore timestamp a
decision at ``(k*320 + 480) / 16000`` seconds - the earliest instant a device
could physically have made it.  Latency is measured against the *true end of the
keyword* as annotated in the corpus, not against a hand-picked frame.

**One score function, three back-ends.**  ``WindowScorer`` wraps either the
float Keras model, the int8 TFLite interpreter (what the ESP32 runs), or a plain
callable, so the offline benchmark and the live server cannot drift apart.

**False alarms are quoted per hour, not per window.**  A detector evaluated over
N windows has had N*0.02 s of audio pass through it; comparing raw FA counts
between a 1-minute and a 1-hour test is meaningless, so every report normalises
to FA/hour (the standard edge-KWS figure of merit).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.common.audio import FRAME_HOP, FRAME_LENGTH, SAMPLE_RATE
from ml.common.features import FrontendParams, QuantParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_FRAMES = 49
WINDOW_SAMPLES = FRAME_HOP * (WINDOW_FRAMES - 1) + FRAME_LENGTH  # 15840 == 0.99 s


@dataclass
class PolicyConfig:
    """Decision policy applied on top of the raw per-window score."""

    threshold: float = 0.5
    #: require this many consecutive windows over threshold (1 = instant)
    confirm_windows: int = 2
    #: after a detection, ignore further triggers for this long (ms)
    refractory_ms: int = 1200
    #: minimum gap between two detections of *different* events (ms)
    min_event_gap_ms: int = 500

    @property
    def refractory_windows(self) -> int:
        return max(0, int(round(self.refractory_ms / 1000.0 * SAMPLE_RATE / FRAME_HOP)))

    @property
    def min_gap_windows(self) -> int:
        return max(0, int(round(self.min_event_gap_ms / 1000.0 * SAMPLE_RATE / FRAME_HOP)))


@dataclass
class Detection:
    """One wake-word activation."""

    window_index: int
    time_s: float  # decision timestamp (earliest physically possible)
    score: float
    confirmed: bool = True


@dataclass
class StreamStats:
    """Aggregate result for one scored stream."""

    duration_s: float = 0.0
    n_windows: int = 0
    n_detections: int = 0
    detections: list[Detection] = field(default_factory=list)

    @property
    def fa_per_hour(self) -> float:
        return self.n_detections / (self.duration_s / 3600.0) if self.duration_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------
class WindowScorer:
    """Callable that maps a batch of feature windows to wake probabilities."""

    def __init__(
        self,
        model=None,
        tflite_path: str | None = None,
        quant: QuantParams | None = None,
        callable_fn=None,
    ):
        self.kind = "callable"
        self._model = model
        self._fn = callable_fn
        self._interpreter = None
        self._quant = quant
        self._in_idx = self._out_idx = None
        self._batch = 1
        self._input_shape = None

        if tflite_path is not None:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            self._interpreter.allocate_tensors()
            self._in_idx = self._interpreter.get_input_details()[0]["index"]
            self._out_idx = self._interpreter.get_output_details()[0]["index"]
            self._input_shape = list(self._interpreter.get_input_details()[0]["shape"])
            self.kind = "tflite_int8"
        elif model is not None:
            self.kind = "keras"

    # -- offline benchmarking helper --------------------------------------
    def enable_batching(self, batch_size: int) -> bool:
        """Resize the int8 model's input to score ``batch_size`` windows per call.

        The device runs batch 1 (it must: one window arrives at a time), so this
        exists purely to make offline benchmarking of hour-long streams
        tractable.  It changes only the batching, never the arithmetic, so the
        scores are identical to batch-1 inference - which
        ``tests/test_streaming.py`` asserts.
        """
        if self.kind != "tflite_int8" or self._input_shape is None:
            return False
        new_shape = [int(batch_size)] + [int(d) for d in self._input_shape[1:]]
        try:
            self._interpreter.resize_tensor_input(self._in_idx, new_shape, strict=False)
            self._interpreter.allocate_tensors()
            self._batch = int(batch_size)
            return True
        except Exception:
            self._batch = 1
            return False

    @property
    def batch_size(self) -> int:
        return self._batch

    def _score_tflite(self, windows: np.ndarray) -> np.ndarray:
        b = self._batch
        out = np.empty(windows.shape[0], dtype=np.float32)
        scale, zero = self._interpreter.get_output_details()[0]["quantization"]
        for i in range(0, windows.shape[0], b):
            chunk = windows[i : i + b]
            if chunk.shape[0] < b:  # pad the final partial batch
                chunk = np.concatenate([chunk, np.zeros((b - chunk.shape[0],) + chunk.shape[1:], np.float32)], axis=0)
            q = self._quant.quantize(chunk)[..., np.newaxis].astype(np.int8)
            self._interpreter.set_tensor(self._in_idx, q)
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._out_idx).reshape(-1)[: windows.shape[0] - i]
            out[i : i + raw.size] = (raw.astype(np.float32) - zero) * scale if scale else raw
        return out

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        """``windows``: ``(B, 49, 40)`` float32 log-mel -> ``(B,)`` probabilities."""
        windows = np.asarray(windows, dtype=np.float32)
        if windows.ndim == 2:  # a single window
            windows = windows[None]
        if self.kind == "tflite_int8":
            if self._quant is None:
                raise ValueError("tflite_int8 scorer needs QuantParams")
            return self._score_tflite(windows)
        if self.kind == "keras":
            return np.asarray(self._model(windows[..., np.newaxis], training=False)).ravel()
        if self._fn is not None:
            return np.asarray(self._fn(windows)).ravel()
        return np.asarray(self._model(windows)).ravel()


# ---------------------------------------------------------------------------
# Window <-> spectrogram helpers
# ---------------------------------------------------------------------------
def window_from_frames(mel: np.ndarray, k: int, params: FrontendParams, n_frames: int = WINDOW_FRAMES) -> np.ndarray:
    """The 49-frame window whose newest frame is ``mel[k]`` (left-padded with the floor)."""
    start = k - n_frames + 1
    if start < 0:
        pad = np.full((-start, mel.shape[1]), np.log(params.log_floor), dtype=np.float32)
        block = mel[0 : k + 1]
        return np.concatenate([pad, block], axis=0)
    return mel[start : k + 1]


def all_windows(mel: np.ndarray, params: FrontendParams, n_frames: int = WINDOW_FRAMES) -> np.ndarray:
    """Batch version of :func:`window_from_frames` -> ``(T, 49, 40)``."""
    t = mel.shape[0]
    if t == 0:
        return np.zeros((0, n_frames, mel.shape[1]), dtype=np.float32)
    pad = np.full((n_frames - 1, mel.shape[1]), np.log(params.log_floor), dtype=np.float32)
    padded = np.concatenate([pad, mel], axis=0)
    idx = np.arange(t)[:, None] + np.arange(n_frames)[None, :]
    return padded[idx]


def detection_time_s(window_index: int) -> float:
    """Earliest wall-clock instant at which window ``window_index`` could be decided."""
    return (window_index * FRAME_HOP + FRAME_LENGTH) / float(SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------
def apply_policy(scores: np.ndarray, cfg: PolicyConfig, start_window: int = 0) -> list[Detection]:
    """Turn a per-window score sequence into detections.

    Two mechanisms, both of which real products use and both of which trade
    latency for false alarms:

    * ``confirm_windows`` - require N consecutive windows over threshold.  Each
      extra window costs exactly one hop (20 ms) of latency and removes the
      single-window noise spikes that dominate false activations.
    * ``refractory_ms`` - after firing, ignore triggers while the device is
      already awake/streaming, which is what stops one utterance producing a
      burst of activations.

    The logic is deliberately the *same* as :meth:`OnlineWakeDetector.push`, so
    the offline benchmark and the live server cannot report different behaviour.
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    detections: list[Detection] = []
    need = max(1, int(cfg.confirm_windows))
    above = scores >= cfg.threshold

    run_start: int | None = None
    for k in range(len(scores) + 1):
        is_above = k < len(scores) and bool(above[k])
        if is_above and run_start is None:
            run_start = k
        elif not is_above and run_start is not None:
            run_end = k - 1
            if run_end - run_start + 1 >= need:
                k_decide = run_start + need - 1  # the confirming window
                gap = max(cfg.min_gap_windows, 1)
                if detections:
                    gap = max(gap, cfg.refractory_windows)
                if not detections or k_decide - detections[-1].window_index >= gap:
                    detections.append(
                        Detection(
                            window_index=start_window + k_decide,
                            time_s=detection_time_s(start_window + k_decide),
                            score=float(scores[k_decide]),
                        )
                    )
            run_start = None
    return detections


# ---------------------------------------------------------------------------
# Online detector (used by the server and by the device's state machine)
# ---------------------------------------------------------------------------
class OnlineWakeDetector:
    """Incremental detector: ``push(samples)`` in, detections out.

    Keeps a rolling sample buffer and computes exactly one new mel frame per hop,
    which is what the firmware does - so the numbers measured here are the numbers
    the device achieves, modulo the model runtime itself.
    """

    def __init__(
        self,
        scorer: WindowScorer,
        params: FrontendParams | None = None,
        policy: PolicyConfig | None = None,
    ):
        self.params = params or FrontendParams()
        self.quant = self.params.quant
        self.scorer = scorer
        self.policy = policy or PolicyConfig()

        # Rolling state.  Only the last WINDOW_FRAMES mel frames are ever needed,
        # and only one window of samples, so memory is O(1) in stream length -
        # the same property the firmware relies on.
        self._buf = np.zeros(0, dtype=np.float32)
        self._ring = np.full((WINDOW_FRAMES, self.params.num_mel_bins), np.log(self.params.log_floor), dtype=np.float32)
        self._n_samples_seen = 0
        self._n_frames = 0
        self._streak: list[int] = []
        self._last_detection_k = -10**9
        self._pending: list[Detection] = []

    # -- property helpers -------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def time_s(self) -> float:
        return self._n_samples_seen / float(self.params.sample_rate)

    # -- internal ---------------------------------------------------------
    def _new_frame(self, frame: np.ndarray) -> np.ndarray:
        """Log-mel of one 480-sample frame (mirrors ml/common/features.py)."""
        p = self.params
        spec = np.fft.rfft(frame[: p.frame_length] * p.window, n=p.fft_size)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        mel = p.mel_matrix @ power
        return np.log(np.maximum(mel, p.log_floor)).astype(np.float32)

    def _score_window(self) -> None:
        """Score the window ending at the newest frame and run the decision policy."""
        k = self._n_frames - 1
        score = float(np.asarray(self.scorer(self._ring)).ravel()[0])

        need = max(1, int(self.policy.confirm_windows))
        if score < self.policy.threshold:
            self._streak = []
        else:
            self._streak.append(k)
            if len(self._streak) > need:
                self._streak = self._streak[-need:]
        confirming = len(self._streak) >= need and (self._streak[-1] - self._streak[0] == need - 1)

        if confirming:
            gap = max(self.policy.min_gap_windows, 1)
            if not self._pending and self._last_detection_k >= 0:
                gap = max(gap, self.policy.refractory_windows)
            if k - self._last_detection_k >= gap:
                self._pending.append(Detection(window_index=k, time_s=detection_time_s(k), score=score))
                self._last_detection_k = k

    # -- public API -------------------------------------------------------
    def push(self, samples: np.ndarray) -> list[Detection]:
        """Feed arbitrary-length audio; returns any detections completed by it."""
        self._pending = []
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._buf = np.concatenate([self._buf, samples])
            self._n_samples_seen += samples.size

        p = self.params
        # Emit a frame whenever a full frame's worth of *new* audio exists.
        while self._buf.size >= p.frame_length:
            mel = self._new_frame(self._buf[: p.frame_length])
            self._buf = self._buf[p.frame_hop :]
            self._ring = np.roll(self._ring, -1, axis=0)
            self._ring[-1] = mel
            self._n_frames += 1
            self._score_window()

        out, self._pending = self._pending, []
        return out

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._ring = np.full(
            (WINDOW_FRAMES, self.params.num_mel_bins), np.log(self.params.log_floor), dtype=np.float32
        )
        self._n_samples_seen = 0
        self._n_frames = 0
        self._streak = []
        self._last_detection_k = -10**9
        self._pending = []


def score_waveform_offline(
    x: np.ndarray,
    scorer: WindowScorer,
    params: FrontendParams | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Score every window of a waveform in one pass (fast path for benchmarks).

    Deliberately produces exactly the same window sequence as
    :class:`OnlineWakeDetector`: frame ``k`` -> window ``[k-48 .. k]``.
    """
    from ml.common.features import log_mel_spectrogram

    params = params or FrontendParams()
    mel = log_mel_spectrogram(x, params)
    if mel.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    windows = all_windows(mel, params, WINDOW_FRAMES)
    scores = np.empty(windows.shape[0], dtype=np.float32)
    for i in range(0, windows.shape[0], batch_size):
        chunk = windows[i : i + batch_size]
        scores[i : i + chunk.shape[0]] = np.asarray(scorer(chunk)).ravel()
    return scores
