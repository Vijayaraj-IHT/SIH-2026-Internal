"""Sliding-window scoring and the wake decision policy.

Two things are being protected here:

1. **The policy's timing arithmetic.** ``detection_time_s`` is the number that
   goes into the latency claim in the write-up, and a wake word that fires 20 ms
   early/late is a difference the reviewers will see on the video.  The formula is
   derived, not tuned: window ``k`` ends at sample ``k*hop + frame_length``.

2. **Online/offline equivalence.** The benchmark scores whole files in one pass;
   the device (and the server's websocket path) processes audio in arbitrary
   chunks.  If those two disagree, every benchmark number is fiction.  The test
   feeds the same audio to both in differently-sized chunks and demands identical
   detections.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.common.features import FrontendParams
from ml.common.streaming import (
    FRAME_HOP,
    FRAME_LENGTH,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    Detection,
    OnlineWakeDetector,
    PolicyConfig,
    WindowScorer,
    all_windows,
    apply_policy,
    detection_time_s,
    score_waveform_offline,
)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def test_detection_time_matches_the_frame_geometry() -> None:
    # window k covers samples [k*hop, k*hop + frame_length) of *its own* newest
    # frame, i.e. it cannot be decided before the end of frame k
    for k in (0, 1, 10, 49, 1000):
        expected = (k * FRAME_HOP + FRAME_LENGTH) / SAMPLE_RATE
        assert detection_time_s(k) == pytest.approx(expected, abs=1e-9)
    # first possible decision is at 30 ms (one frame), each hop adds exactly 20 ms
    assert detection_time_s(0) == pytest.approx(0.030)
    assert detection_time_s(1) - detection_time_s(0) == pytest.approx(0.020)
    assert detection_time_s(10) == pytest.approx(0.23)


def test_window_geometry_is_49_frames() -> None:
    mel = np.random.default_rng(0).standard_normal((60, 40)).astype(np.float32)
    windows = all_windows(mel, FrontendParams())
    assert windows.shape == (60, WINDOW_FRAMES, 40)
    # the newest frame of window k is mel[k]
    assert np.allclose(windows[20, -1], mel[20])


def test_window_padding_uses_the_log_floor() -> None:
    params = FrontendParams()
    mel = np.full((10, params.num_mel_bins), -5.0, dtype=np.float32)
    windows = all_windows(mel, params)
    # window 0 has 48 padding frames on the left
    floor = np.log(params.log_floor)
    assert np.allclose(windows[0, : WINDOW_FRAMES - 1], floor, atol=1e-5)
    assert np.allclose(windows[0, -1], -5.0)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def test_confirm_windows_rejects_a_single_spike() -> None:
    scores = np.zeros(50, dtype=np.float32)
    scores[10] = 0.99
    assert apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=1)) != []
    assert apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=2)) == []


def test_confirm_windows_fires_on_a_run_and_reports_the_confirming_window() -> None:
    scores = np.zeros(50, dtype=np.float32)
    scores[10:14] = 0.9  # four consecutive windows
    (det,) = apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=3))
    # the decision is taken on the 3rd window of the run, not the 4th: that is the
    # latency floor the policy imposes, and it must not silently grow
    assert det.window_index == 12
    assert det.time_s == pytest.approx(detection_time_s(12))


def test_extra_confirmation_costs_exactly_one_hop_each() -> None:
    scores = np.zeros(80, dtype=np.float32)
    scores[20:30] = 0.9
    times = [apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=n))[0].time_s for n in (1, 2, 3)]
    assert times[1] - times[0] == pytest.approx(0.020)
    assert times[2] - times[1] == pytest.approx(0.020)


def test_threshold_is_inclusive() -> None:
    scores = np.full(10, 0.5, dtype=np.float32)
    assert apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=1)) != []
    assert apply_policy(scores, PolicyConfig(threshold=0.51, confirm_windows=1)) == []


def test_refractory_suppresses_a_second_burst() -> None:
    scores = np.zeros(200, dtype=np.float32)
    scores[10:14] = 0.9
    scores[30:34] = 0.9  # 20 windows later = 400 ms, inside the 1200 ms refractory
    cfg = PolicyConfig(threshold=0.5, confirm_windows=2, refractory_ms=1200)
    assert len(apply_policy(scores, cfg)) == 1


def test_detection_after_the_refractory_is_kept() -> None:
    scores = np.zeros(400, dtype=np.float32)
    scores[10:14] = 0.9
    scores[100:104] = 0.9  # 90 windows = 1.8 s later, outside the refractory
    cfg = PolicyConfig(threshold=0.5, confirm_windows=2, refractory_ms=1200)
    dets = apply_policy(scores, cfg)
    assert len(dets) == 2
    assert dets[1].window_index - dets[0].window_index == pytest.approx(90, abs=2)


def test_min_event_gap_applies_between_distinct_events() -> None:
    scores = np.zeros(300, dtype=np.float32)
    scores[10:14] = 0.9  # decides at window 11
    scores[18:22] = 0.9  # would decide at window 19: only 160 ms later
    cfg = PolicyConfig(threshold=0.5, confirm_windows=2, refractory_ms=0, min_event_gap_ms=500)
    gap_windows = int(0.5 * SAMPLE_RATE / FRAME_HOP)
    assert gap_windows == 25
    dets = apply_policy(scores, cfg)
    assert len(dets) == 1, "a second activation inside the minimum gap must be suppressed"
    # ... and an activation beyond the gap survives
    scores[60:64] = 0.9  # decides at window 61: 50 windows = 1 s after the first
    dets = apply_policy(scores, cfg)
    assert len(dets) == 2
    assert dets[1].window_index - dets[0].window_index >= gap_windows


def test_policy_window_arithmetic() -> None:
    cfg = PolicyConfig(refractory_ms=1200, min_event_gap_ms=500)
    assert cfg.refractory_windows == int(1.2 * SAMPLE_RATE / FRAME_HOP)
    assert cfg.min_gap_windows == int(0.5 * SAMPLE_RATE / FRAME_HOP)


# ---------------------------------------------------------------------------
# Online detector
# ---------------------------------------------------------------------------
class _FakeScorer:
    """Scores a window by its mean energy relative to a fixed reference.

    Cheap, deterministic and - crucially - dependent on the *whole* window, so a
    framing mistake in the online detector changes the score.
    """

    def __init__(self, level: float = -10.0):
        self.level = level
        self.calls = 0

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        self.calls += 1
        w = np.asarray(windows, dtype=np.float32)
        if w.ndim == 2:
            w = w[None]
        return 1.0 / (1.0 + np.exp(-(w.mean(axis=(1, 2)) - self.level)))


def _tone_bursts(n: int, bursts: list[tuple[int, int]], sr: int = SAMPLE_RATE) -> np.ndarray:
    x = 1e-4 * np.random.default_rng(0).standard_normal(n).astype(np.float32)
    for start, length in bursts:
        k = np.arange(length)
        x[start : start + length] += 0.4 * np.sin(2 * np.pi * 440 * k / sr).astype(np.float32)
    return x


def test_online_and_offline_agree_on_the_same_audio() -> None:
    params = FrontendParams()
    x = _tone_bursts(SAMPLE_RATE * 4, [(4000, 3000), (30000, 3000), (50000, 3000)])
    scorer_offline = WindowScorer(callable_fn=_FakeScorer())
    offline_scores = score_waveform_offline(x, scorer_offline, params)

    scorer_online = WindowScorer(callable_fn=_FakeScorer())
    det = OnlineWakeDetector(scorer_online, params, PolicyConfig(threshold=0.5, confirm_windows=2))
    online_dets: list[Detection] = []
    for i in range(0, x.size, 1600):  # feed in 100 ms chunks
        online_dets.extend(det.push(x[i : i + 1600]))

    cfg = PolicyConfig(threshold=0.5, confirm_windows=2)
    offline_dets = apply_policy(offline_scores, cfg)
    assert len(offline_scores) == det.n_frames, "online and offline framing disagree"
    assert [d.window_index for d in online_dets] == [d.window_index for d in offline_dets]
    for a, b in zip(online_dets, offline_dets):
        assert a.time_s == pytest.approx(b.time_s)
        assert a.score == pytest.approx(b.score, abs=1e-5)


def test_online_detector_is_chunk_size_invariant() -> None:
    params = FrontendParams()
    x = _tone_bursts(SAMPLE_RATE * 3, [(2000, 2500), (20000, 2500)])
    results = []
    for chunk in (320, 480, 1600, 8_000):
        det = OnlineWakeDetector(WindowScorer(callable_fn=_FakeScorer()), params, PolicyConfig(threshold=0.5))
        dets: list[Detection] = []
        for i in range(0, x.size, chunk):
            dets.extend(det.push(x[i : i + chunk]))
        results.append([d.window_index for d in dets])
    assert all(r == results[0] for r in results), results


def test_online_detector_frame_count() -> None:
    params = FrontendParams()
    det = OnlineWakeDetector(WindowScorer(callable_fn=_FakeScorer()), params)
    n = SAMPLE_RATE // 2  # 500 ms
    det.push(np.zeros(n, dtype=np.float32))
    expected = (n - FRAME_LENGTH) // FRAME_HOP + 1
    assert det.n_frames == expected
    assert det.time_s == pytest.approx(n / SAMPLE_RATE)


def test_online_detector_trims_its_buffer() -> None:
    """The device runs for months: memory must not grow with audio length."""
    params = FrontendParams()
    det = OnlineWakeDetector(WindowScorer(callable_fn=_FakeScorer()), params)
    for _ in range(60):
        det.push(np.zeros(4800, dtype=np.float32))  # 10 s of audio
    assert det._buf.size < FRAME_HOP * 2, "sample buffer grew without bound"
    assert det._ring.shape == (WINDOW_FRAMES, params.num_mel_bins)


def test_reset_clears_state() -> None:
    params = FrontendParams()
    x = _tone_bursts(SAMPLE_RATE * 2, [(1000, 2000)])
    det = OnlineWakeDetector(WindowScorer(callable_fn=_FakeScorer()), params, PolicyConfig(threshold=0.5))
    det.push(x)
    assert det.n_frames > 0
    det.reset()
    assert det.n_frames == 0 and det.time_s == 0.0
    after = det.push(x)
    assert [d.window_index for d in after] == [
        d.window_index
        for d in OnlineWakeDetector(
            WindowScorer(callable_fn=_FakeScorer()), params, PolicyConfig(threshold=0.5)
        ).push(x)
    ]


def test_silence_never_fires() -> None:
    params = FrontendParams()
    x = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    scores = score_waveform_offline(x, WindowScorer(callable_fn=_FakeScorer()), params)
    assert np.isfinite(scores).all()
    assert apply_policy(scores, PolicyConfig(threshold=0.5, confirm_windows=1)) == []


def test_fa_rate_arithmetic() -> None:
    from ml.common.streaming import StreamStats

    stats = StreamStats(duration_s=1800.0, n_windows=1000, n_detections=3)
    assert stats.fa_per_hour == pytest.approx(6.0)
