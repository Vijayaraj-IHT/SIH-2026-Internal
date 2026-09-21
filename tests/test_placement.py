"""Window placement and labelling.

The placement function decides what "positive" means for the entire project, so
it is tested against the geometry it claims to implement rather than against a
snapshot of its own output.

The claim, derived from ``w = c + end_pos - offset``:

* the visible fraction of the utterance is ``min(1, end_pos / kw_len)``,
* a window is positive iff the keyword ends inside it **and** at least
  ``MIN_KEYWORD_VISIBLE`` of the keyword is visible,
* everything else - an unknown word, a background crop, a keyword that has not
  finished yet - is a negative.

A previous version bounded ``end_pos`` by the head crop, which pushed late
utterances past the window edge and silently mislabelled a large share of the
positive class. These tests exist so that cannot happen again quietly.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from ml.common.features import FrontendParams
from ml.data.pipeline import (
    MIN_KEYWORD_VISIBLE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    AugmentConfig,
    TFFrontend,
    place_clip,
)


def _frontend() -> TFFrontend:
    return TFFrontend(FrontendParams())


def _clip(n: int, onset: int, offset: int, seed: int = 0) -> np.ndarray:
    """Speech-like burst over a quiet floor, with a known [onset, offset)."""
    rng = np.random.default_rng(seed)
    x = 0.002 * rng.standard_normal(n).astype(np.float32)
    k = np.arange(max(0, offset - onset))
    x[onset:offset] += (0.5 * np.sin(2 * np.pi * 220 * k / 16000.0)).astype(np.float32)
    return x


def _place(x, onset, offset, kind="keyword", *, train=False, slot=0, cfg=None, seed=0, path=None):
    cfg = cfg or AugmentConfig()
    rng = tf.random.Generator.from_seed(seed)
    return place_clip(
        tf.constant(np.asarray(x), tf.float32),
        tf.constant(onset, tf.int32),
        tf.constant(offset, tf.int32),
        tf.constant(kind),
        _frontend(),
        cfg,
        rng,
        train,
        path=tf.constant(path or f"clips/{kind}/{onset}_{offset}.wav"),
        slot=tf.constant(slot, tf.int32),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("onset,offset", [(0, 8000), (5000, 14000), (15000, 21000), (24000, 30000)])
def test_positive_windows_show_enough_of_the_keyword(onset: int, offset: int) -> None:
    n = max(offset + 6000, 40000)
    x = _clip(n, onset, offset)
    positives = 0
    for slot in range(16):
        _f, wake, end_frame, _fl = _place(x, onset, offset, "keyword", slot=slot)
        if float(wake) != 1.0:
            continue
        positives += 1
        end_pos = int(end_frame) * WINDOW_SAMPLES // WINDOW_FRAMES
        assert 0 <= end_pos < WINDOW_SAMPLES, "a positive window must contain the end of the keyword"
        visible = min(1.0, end_pos / max(1, offset - onset))
        assert visible >= MIN_KEYWORD_VISIBLE - 0.06, f"only {visible:.2f} of the keyword is audible"
    # Half the windows of a clip are the "keyword has not finished" case, so with
    # sixteen windows a clip must contribute several positives.  A single clip can
    # legitimately be all-partial over four slots, which is why this counts over
    # more slots than the evaluation set uses.
    assert positives >= 3, f"only {positives}/16 windows of the clip were positive"


@pytest.mark.parametrize("onset,offset", [(0, 7000), (6000, 13000), (18000, 24000)])
def test_visibility_bound_matches_brute_force(onset: int, offset: int) -> None:
    """``end_pos >= MIN_KEYWORD_VISIBLE * kw_len`` must equal "enough is audible"."""
    kw_len = offset - onset
    for end_pos in range(0, WINDOW_SAMPLES, 640):
        visible = min(end_pos, kw_len) / kw_len
        assert (visible >= MIN_KEYWORD_VISIBLE) == (end_pos >= MIN_KEYWORD_VISIBLE * kw_len)


def test_keyword_clips_produce_both_cases() -> None:
    """Roughly half the evaluation windows are "not finished yet" negatives.

    That balance is the point of the evaluation set: it measures the recall the
    detector should have and the premature firing it should not, on every run.
    """
    onset, offset = 8000, 15000
    x = _clip(40000, onset, offset)
    labels = [
        float(_place(x, onset, offset, "keyword", slot=s, path="clips/kw/a.wav")[1])
        for s in range(16)
    ]
    assert 1.0 in labels and 0.0 in labels, labels
    share = sum(labels) / len(labels)
    assert 0.25 <= share <= 0.75, f"evaluation windows are not balanced: {share:.2f} positive"


def test_unknown_words_are_always_negative() -> None:
    onset, offset = 5000, 13000
    x = _clip(30000, onset, offset)
    for kind in ("fuzzy", "gsc_word", "gsc_other"):
        for slot in range(4):
            _f, wake, end_frame, _fl = _place(x, onset, offset, kind, slot=slot)
            assert float(wake) == 0.0, f"{kind} slot {slot} was labelled positive"
            assert int(end_frame) == -1


def test_background_clips_are_negative_and_labelled_so() -> None:
    x = np.random.default_rng(0).standard_normal(48000).astype(np.float32) * 0.1
    for slot in range(4):
        _f, wake, end_frame, frame_labels = _place(x, 0, 48000, "background", slot=slot)
        assert float(wake) == 0.0
        assert int(end_frame) == -1
        assert not np.any(np.asarray(frame_labels))


# ---------------------------------------------------------------------------
# Determinism (this is what makes checkpoint selection meaningful)
# ---------------------------------------------------------------------------
def test_eval_placement_is_deterministic() -> None:
    onset, offset = 6000, 15000
    x = _clip(30000, onset, offset)
    a = _place(x, onset, offset, "keyword", slot=2, path="clips/kw/a.wav")
    b = _place(x, onset, offset, "keyword", slot=2, path="clips/kw/a.wav")
    assert np.array_equal(a[0].numpy(), b[0].numpy())
    assert (float(a[1]), int(a[2])) == (float(b[1]), int(b[2]))


def test_eval_slots_are_not_all_the_same_window() -> None:
    onset, offset = 6000, 15000
    x = _clip(30000, onset, offset)
    windows = [_place(x, onset, offset, "keyword", slot=s)[0].numpy() for s in range(4)]
    assert not np.array_equal(windows[0], windows[1]), "the four evaluation windows must differ"


def test_different_clips_get_different_placements() -> None:
    onset, offset = 6000, 15000
    x = _clip(30000, onset, offset)
    a = _place(x, onset, offset, "keyword", slot=0, path="clips/kw/a.wav")[0].numpy()
    b = _place(x, onset, offset, "keyword", slot=0, path="clips/kw/zzz.wav")[0].numpy()
    assert not np.array_equal(a, b)


def test_training_placement_is_random() -> None:
    onset, offset = 6000, 15000
    x = _clip(30000, onset, offset)
    rng = tf.random.Generator.from_seed(0)
    seen = set()
    for _ in range(16):
        out = place_clip(
            tf.constant(x, tf.float32),
            tf.constant(onset, tf.int32),
            tf.constant(offset, tf.int32),
            tf.constant("keyword"),
            _frontend(),
            AugmentConfig(),
            rng,
            True,
            path=tf.constant("clips/kw/a.wav"),
            slot=tf.constant(0, tf.int32),
        )
        seen.add(int(out[2].numpy()))
    assert len(seen) > 3, f"training placement did not vary: {seen}"


# ---------------------------------------------------------------------------
# Frame-level auxiliary targets
# ---------------------------------------------------------------------------
def test_frame_labels_are_a_contiguous_run() -> None:
    onset, offset = 5000, 12000
    x = _clip(30000, onset, offset)
    for slot in range(4):
        _f, wake, _ef, labels = _place(x, onset, offset, "keyword", slot=slot)
        labels = np.asarray(labels)
        assert labels.shape == (7,)
        assert set(np.unique(labels)).issubset({0.0, 1.0})
        if float(wake) == 1.0 and labels.any():
            idx = np.flatnonzero(labels == 1.0)
            assert np.all(np.diff(idx) == 1), f"frame targets are not contiguous: {idx}"
