"""Streaming keyword-spotting input pipeline: framing, augmentation, features.

Everything that touches a waveform during training lives here, and it is
expressed purely in TensorFlow ops so it runs inside ``tf.data`` without Python
callbacks.  ``tests/test_pipeline_parity.py`` asserts that the log-mel it
produces matches ``ml/common/features.py`` (the reference numpy path that the C
firmware is checked against).

The single most important idea in this file
-------------------------------------------
A 0.99 s-window wake word detector must answer four different situations
correctly, and the *only* way to get that is to build all four into the label
definition:

===============  ==================================================  =======
situation        meaning                                             label
===============  ==================================================  =======
end inside       the keyword (or a confusable) finishes in window     1/0
end not arrived  only the first part of the keyword is audible          0
starts after     the window ends before the keyword begins             0
background       speech-free window                                    0
===============  ==================================================  =======

The "end not arrived" case is what stops the classic failure mode where a
detector fires on the first phoneme ("hoping to be fast") and the reported
latency becomes fiction.  Because every window in which the keyword has not yet
finished is a *negative*, the earliest possible detection is the first window
that actually contains the end of the keyword - so latency is measured against
a physically meaningful event.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.common.audio import FRAME_HOP, FRAME_LENGTH, SAMPLE_RATE
from ml.common.features import CONTEXT_FRAMES, FrontendParams, mel_filterbank, hann_periodic

WINDOW_FRAMES = CONTEXT_FRAMES  # 49
WINDOW_SAMPLES = FRAME_HOP * (WINDOW_FRAMES - 1) + FRAME_LENGTH  # 15840 == 0.99 s
NOISE_BANK_SECONDS = 3.0
NOISE_BANK_SAMPLES = int(NOISE_BANK_SECONDS * SAMPLE_RATE)

#: A positive window must show at least this fraction of the keyword, otherwise
#: the head crop has removed so much that the example is not the word any more.
MIN_KEYWORD_VISIBLE = 0.35
#: Hardest negative class: the keyword end is just outside the window.  Sampling
#: the end position up to this far beyond the window edge models "the user is
#: 150 ms away from finishing the word".
PARTIAL_LOOKAHEAD_FRACTION = 0.9
#: Windows scored per clip in evaluation (half "end inside", half "not finished").
EVAL_SLOTS = 4

#: Shuffle buffer for the training set.  It must exceed the size of the largest
#: same-class block in the TFRecord shards (the writer emits every keyword clip,
#: then every fuzzy negative, then the Speech Commands clips), otherwise batches
#: arrive sorted by class and batch normalisation silently becomes per-class
#: normalisation.  A 1,000-element buffer was the original bug here.
SHUFFLE_BUFFER = 40_000


@dataclass
class AugmentConfig:
    """Waveform-domain augmentation knobs (all no-ops for validation/test)."""

    gain_db: float = 6.0
    noise_prob: float = 0.7
    noise_snr_db: tuple[float, float] = (0.0, 20.0)
    #: probability that a keyword clip is turned into a "not finished yet" negative
    partial_negative_prob: float = 0.30
    #: probability of an all-silence window
    silence_prob: float = 0.03
    #: probability of a purely random crop from a background recording
    background_prob: float = 0.05


# ---------------------------------------------------------------------------
# TF-side front-end (mirrors ml/common/features.py)
# ---------------------------------------------------------------------------
class TFFrontend:
    """Log-mel extraction as graph ops, with the constants baked in."""

    def __init__(self, params: FrontendParams | None = None):
        self.params = params or FrontendParams()
        p = self.params
        self._window = tf.constant(p.window, dtype=tf.float32)
        self._mel = tf.constant(p.mel_matrix.T, dtype=tf.float32)  # (n_freqs, n_mels)

    def batch_logmel(self, waveforms: tf.Tensor) -> tf.Tensor:
        """``(B, N)`` float32 waveforms -> ``(B, T, n_mels)`` log-mel."""
        p = self.params
        frames = tf.signal.frame(
            waveforms,
            frame_length=p.frame_length,
            frame_step=p.frame_hop,
            pad_end=False,
            axis=1,
        )  # (B, T, frame_length)
        windowed = frames * self._window
        spec = tf.signal.rfft(windowed, fft_length=[p.fft_size])
        power = tf.square(tf.math.real(spec)) + tf.square(tf.math.imag(spec))
        mel = tf.matmul(power, self._mel)
        return tf.math.log(tf.maximum(mel, p.log_floor))

    def pad_to_context(self, logmel: tf.Tensor) -> tf.Tensor:
        """Left-pad with the log-floor value so the time axis is exactly 49."""
        t = tf.shape(logmel)[1]
        pad = tf.maximum(WINDOW_FRAMES - t, 0)
        padded = tf.pad(logmel, [[0, 0], [pad, 0], [0, 0]], constant_values=tf.math.log(self.params.log_floor))
        return padded[:, :WINDOW_FRAMES, :]


# ---------------------------------------------------------------------------
# Record decoding
# ---------------------------------------------------------------------------
def decode_record(example: tf.Tensor) -> dict:
    """TFRecord -> dict with a float32 waveform and its metadata."""
    from ml.data.build_cache import FEATURE_SPEC

    feat = tf.io.parse_single_example(example, FEATURE_SPEC)
    audio = tf.io.decode_raw(feat["audio"], tf.int16)
    x = tf.cast(audio, tf.float32) / 32768.0
    return {
        "x": x,
        "label": tf.cast(feat["label"], tf.int32),
        "kind": feat["kind"],
        "speaker": feat["speaker"],
        "source": feat["source"],
        "detail": feat["detail"],
        "split": feat["split"],
        "onset": tf.cast(feat["onset"], tf.int32),
        "offset": tf.cast(feat["offset"], tf.int32),
        "path": feat["path"],
    }


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def random_crop_or_pad(x: tf.Tensor, target: tf.Tensor, rng: tf.random.Generator) -> tf.Tensor:
    """Return exactly ``target`` samples taken from a random position in ``x``."""
    n = tf.shape(x)[0]

    def do_pad() -> tf.Tensor:
        return tf.pad(x, [[0, target - n]])

    def do_crop() -> tf.Tensor:
        start = rng.uniform([], 0, tf.cast(n - target + 1, tf.float32), dtype=tf.float32)
        return tf.slice(x, [tf.cast(start, tf.int32)], [target])

    return tf.cond(n >= target, do_crop, do_pad)


def mix_noise(
    x: tf.Tensor,
    noise_bank: tf.Tensor,
    rng: tf.random.Generator,
    snr_db_range: tuple[float, float],
) -> tf.Tensor:
    """Add a randomly cropped real background recording at a random SNR."""
    idx = rng.uniform([], 0, tf.cast(tf.shape(noise_bank)[0], tf.float32), dtype=tf.float32)
    noise = tf.gather(noise_bank, tf.cast(idx, tf.int32))
    noise = random_crop_or_pad(noise, tf.shape(x)[0], rng)

    def rms(t: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(tf.reduce_mean(tf.square(t)) + 1e-12)

    snr = rng.uniform([], snr_db_range[0], snr_db_range[1], dtype=tf.float32)
    # scale noise so that speech_rms / noise_rms == 10^(snr/20)
    gain = rms(x) / (rms(noise) * tf.pow(10.0, snr / 20.0) + 1e-12)
    return x + noise * gain


def augment_clip(x: tf.Tensor, noise_bank: tf.Tensor, cfg: AugmentConfig, rng: tf.random.Generator) -> tf.Tensor:
    """Gain jitter + real background-noise mixing (training only)."""
    gain_db = rng.uniform([], -cfg.gain_db, cfg.gain_db, dtype=tf.float32)
    x = x * tf.pow(10.0, gain_db / 20.0)

    def add_noise() -> tf.Tensor:
        return mix_noise(x, noise_bank, rng, cfg.noise_snr_db)

    x = tf.cond(rng.uniform([], 0.0, 1.0) < cfg.noise_prob, add_noise, lambda: x)
    return tf.clip_by_value(x, -1.0, 0.999969)


# ---------------------------------------------------------------------------
# Placement: variable-length clip -> fixed 0.99 s (49-frame) window + labels
# ---------------------------------------------------------------------------
def place_clip(
    x: tf.Tensor,
    onset: tf.Tensor,
    offset: tf.Tensor,
    kind: tf.Tensor,
    frontend: TFFrontend,
    cfg: AugmentConfig,
    rng: tf.random.Generator,
    train: bool,
    path: tf.Tensor | None = None,
    slot: tf.Tensor | None = None,
    noise_bank: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Build one analysis window plus its labels.

    Returns ``(features, wake_label, end_frame, frame_labels)``:

    * ``features``     - ``(49, 40)`` float32 log-mel,
    * ``wake_label``   - scalar 1.0/0.0, the deployed head's target,
    * ``end_frame``    - index (0..48) of the frame in which the keyword ends, or
      -1 when it is not visible; the streaming simulator uses this as ground truth,
    * ``frame_labels`` - ``(7,)`` auxiliary per-frame targets.

    Placement geometry (derived, not guessed)
    -----------------------------------------
    Clip sample ``c`` lands at window index ``w = c + end_pos - offset``, so the
    window covers clip samples ``[offset - end_pos, offset - end_pos + W)``.  From
    that single relation:

    * the visible part of the utterance is ``[max(onset, offset - end_pos), offset)``,
    * the visible fraction is ``min(1, end_pos / kw_len)``,
    * the placement is a positive iff the keyword ends inside the window *and*
    enough of it is visible: ``kw_len * MIN_KEYWORD_VISIBLE <= end_pos < W``.

    The second requirement is what stops a window that contains only the last
    phoneme of the keyword from being labelled positive.  An earlier version
    bounded ``end_pos`` by the head crop instead and silently mislabelled roughly
    40% of the positive class; ``tests/test_placement.py`` now checks this against
    a brute-force simulation.

    Determinism in evaluation
    -------------------------
    Training re-samples the offset for every clip on every epoch - that is the
    augmentation.  Evaluation cannot: a validation metric that moves every epoch
    makes checkpoint selection noise, and it hides real regressions.  So in
    evaluation the offset comes from a hash of the clip path plus a caller-supplied
    window index, which decorrelates the windows of one clip while staying
    byte-identical across epochs.
    """
    n = tf.shape(x)[0]
    w = float(WINDOW_SAMPLES)
    kw_len = tf.maximum(tf.cast(offset - onset, tf.float32), 1.0)

    is_keyword = tf.equal(kind, "keyword")
    is_background = tf.equal(kind, "background")
    is_other_word = tf.logical_or(tf.equal(kind, "fuzzy"), tf.strings.regex_full_match(kind, "gsc.*"))

    # ---- deterministic offset in evaluation -----------------------------
    # One unit value per (clip, window index).  to_hash_bucket_fast returns int64;
    # keep the arithmetic in int64 and cast at the end.
    if path is None:
        path = tf.constant("unknown")
    slot_i = tf.constant(0, tf.int64) if slot is None else tf.cast(tf.reshape(slot, []), tf.int64)
    # Hash the path *together with* the window index.  Adding a small multiple of
    # the slot to one hash (the first version did `hash * 7 + slot * 131`) only
    # moves the value by ~0.01% per slot, so all four windows of a clip landed on
    # the same side of every split point and the "four situations per clip" design
    # silently collapsed into "one situation, four times".
    seed_key = tf.strings.join(
        [tf.reshape(path, []), tf.strings.as_string(tf.reshape(slot_i, [])), "kws"], separator="#"
    )
    hashed = tf.cast(tf.strings.to_hash_bucket_fast(seed_key, 1_000_003), tf.int64)
    mixed = (hashed * 2_654_435_761 + 1_013_904_223) % 4_294_967_291
    h_unit = tf.cast(mixed % 1_000_003, tf.float32) / 1_000_003.0

    def draw(lo: tf.Tensor, hi: tf.Tensor) -> tf.Tensor:
        """Uniform on [lo, hi): fresh per epoch in training, path-seeded in eval."""
        if train:
            return rng.uniform([], lo, hi, dtype=tf.float32)
        return lo + (hi - lo) * h_unit

    # ---- the two interesting placements --------------------------------
    bounded_lo = tf.maximum(tf.constant(0.10 * w), MIN_KEYWORD_VISIBLE * kw_len)
    bounded_hi = tf.constant(0.97 * w)
    bounded_hi = tf.maximum(bounded_hi, bounded_lo + 1.0)

    aligned = draw(bounded_lo, bounded_hi)

    partial = draw(
        tf.constant(w + 1.0),
        tf.constant(w) + PARTIAL_LOOKAHEAD_FRACTION * kw_len + 1.0,
    )
    # The two arms must not overlap, otherwise the "partial" draw can land inside
    # the window and be labelled positive after all - which is what happened when
    # the lookahead exceeded the window: the keyword simply ended inside the window
    # and the window was a positive.
    partial = tf.maximum(partial, tf.constant(w + 1.0))

    def pick_end_pos() -> tf.Tensor:
        """Choose between the two placements.

        Training mixes them with ``partial_negative_prob``.  Evaluation uses the
        hash parity instead, so half of every class is scored in the "end inside"
        situation and half in the "not finished yet" situation - the validation set
        then measures both the recall it is supposed to have and the premature fire
        it is not.
        """
        if train:
            use_partial = rng.uniform([], 0.0, 1.0) < cfg.partial_negative_prob
        else:
            use_partial = h_unit > 0.5
        return tf.cond(use_partial, lambda: partial, lambda: aligned)

    end_pos = tf.case(
        [
            (is_background, lambda: tf.constant(-1.0)),
            (is_keyword, pick_end_pos),
            (is_other_word, pick_end_pos),
        ],
        default=lambda: tf.constant(-1.0),
        exclusive=True,
    )

    visible = tf.logical_and(tf.greater_equal(end_pos, 0.0), tf.less(end_pos, tf.cast(w, tf.float32)))
    end_pos_i = tf.cast(tf.floor(end_pos), tf.int32)

    # ---- slice the window ----------------------------------------------
    placed_start = end_pos_i - offset
    pad_left = tf.maximum(0, -placed_start)
    pad_right = tf.maximum(0, placed_start + WINDOW_SAMPLES - n)
    xp = tf.pad(x, [[pad_left, pad_right]])
    start = placed_start + pad_left  # >= 0 by construction
    window = tf.slice(xp, [start], [WINDOW_SAMPLES])

    def random_background() -> tf.Tensor:
        """Background clips carry no keyword: take a random crop instead."""
        if noise_bank is None or tf.shape(noise_bank)[0] == 0:
            return window
        idx = rng.uniform([], 0, tf.cast(tf.shape(noise_bank)[0], tf.float32), dtype=tf.float32)
        clip = tf.gather(noise_bank, tf.cast(idx, tf.int32))
        if train:
            src = n
            off = rng.uniform([], 0.0, tf.cast(tf.maximum(src - WINDOW_SAMPLES + 1, 1), tf.float32),
                              dtype=tf.float32)
            off = tf.cast(tf.maximum(0.0, tf.minimum(off, tf.cast(tf.maximum(src - WINDOW_SAMPLES, 0),
                                                                    tf.float32))), tf.int32)
            return tf.slice(tf.pad(clip, [[0, tf.maximum(0, WINDOW_SAMPLES - tf.shape(clip)[0])]]),
                            [off], [WINDOW_SAMPLES])
        # deterministic in evaluation
        off = tf.cast(h_unit * tf.cast(tf.maximum(tf.shape(clip)[0] - WINDOW_SAMPLES, 0), tf.float32),
                      tf.int32)
        return tf.slice(tf.pad(clip, [[0, tf.maximum(0, WINDOW_SAMPLES - tf.shape(clip)[0])]]),
                        [off], [WINDOW_SAMPLES])

    window = tf.cond(is_background, random_background, lambda: window)

    # ---- features -------------------------------------------------------
    features = frontend.batch_logmel(window[tf.newaxis, :])
    features = frontend.pad_to_context(features)[0]  # (49, 40)

    # ---- labels ---------------------------------------------------------
    wake_label = tf.where(is_keyword, tf.where(visible, 1.0, 0.0), 0.0)
    labelled = tf.logical_and(is_keyword, visible)
    end_frame = tf.where(labelled, end_pos_i * WINDOW_FRAMES // WINDOW_SAMPLES, tf.constant(-1))
    frame_labels = make_frame_labels(labelled, end_frame, onset, offset, end_pos_i, is_keyword)

    return features, wake_label, tf.cast(end_frame, tf.int32), frame_labels


def make_frame_labels(
    visible: tf.Tensor,
    end_frame: tf.Tensor,
    onset: tf.Tensor,
    offset: tf.Tensor,
    end_pos_i: tf.Tensor,
    is_keyword: tf.Tensor,
) -> tf.Tensor:
    """Auxiliary per-frame targets: 1 for encoder frames covering the keyword."""
    n_out = 7  # encoder time steps: 49 -> 25 -> 13 -> 7
    idx = tf.range(n_out, dtype=tf.float32)
    # Temporal stride of the encoder is 2 (stem) * 2 (block 3) * 2 (block 5) = 8
    # input frames, so encoder step j covers input frames [8j, 8j + 8).
    span = 8.0
    key_start_frame = tf.cast(onset, tf.float32) * float(WINDOW_FRAMES) / float(WINDOW_SAMPLES) + (
        tf.cast(end_pos_i - offset, tf.float32) * float(WINDOW_FRAMES) / float(WINDOW_SAMPLES)
    )
    key_end_frame = tf.cast(end_pos_i, tf.float32) * float(WINDOW_FRAMES) / float(WINDOW_SAMPLES)
    active = tf.logical_and(idx * span >= key_start_frame - span / 2, idx * span <= key_end_frame + span / 2)
    active = tf.logical_and(active, visible)
    active = tf.logical_and(active, is_keyword)
    return tf.cast(active, tf.float32)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def load_noise_bank(cache_dir: Path | str, max_clips: int = 96) -> tf.Tensor:
    """Load the held-out-for-training background recordings into a constant tensor.

    These are real urban recordings (UrbanSound8K excerpts shipped with the
    keyword corpus).  Keeping them in memory as a graph constant lets the
    augmentation happen inside ``tf.data`` with no Python overhead.
    """
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("noise_bank-*.tfrec"))
    clips: list[np.ndarray] = []
    if files:
        spec = None
        for f in files:
            for raw in tf.data.TFRecordDataset([str(f)]):
                if spec is None:
                    from ml.data.build_cache import FEATURE_SPEC

                    spec = FEATURE_SPEC
                feat = tf.io.parse_single_example(raw, spec)
                audio = tf.io.decode_raw(feat["audio"], tf.int16).numpy().astype(np.float32) / 32768.0
                if audio.size < NOISE_BANK_SAMPLES:
                    audio = np.pad(audio, (0, NOISE_BANK_SAMPLES - audio.size))
                else:
                    audio = audio[:NOISE_BANK_SAMPLES]
                clips.append(audio)
                if len(clips) >= max_clips:
                    break
            if len(clips) >= max_clips:
                break
    if not clips:
        # Deterministic synthetic fallback so training never breaks.
        rng = np.random.default_rng(0)
        clips = [rng.standard_normal(NOISE_BANK_SAMPLES).astype(np.float32) * 0.05 for _ in range(8)]
    return tf.constant(np.stack(clips), dtype=tf.float32)


def build_dataset(
    files: list[str],
    frontend: TFFrontend,
    cfg: AugmentConfig,
    batch_size: int,
    noise_bank: tf.Tensor,
    seed: int = 0,
    train: bool = True,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """TFRecord shards -> batched ``(features, {wake, frame})`` dataset.

    In evaluation mode the shards are read ``EVAL_SLOTS`` times, each pass using a
    different fixed window index, which is what makes the validation set both
    larger and deterministic (see :func:`place_clip`).
    """
    files = [str(f) for f in files]
    if not files:
        raise ValueError("no shards supplied")

    rng = tf.random.Generator.from_seed(seed + (1 if train else 0))

    def _map_factory(slot: int):
        def _map(example: tf.Tensor):
            rec = decode_record(example)
            x = rec["x"]
            if train:
                x = augment_clip(x, noise_bank, cfg, rng)
            feats, wake, _end_frame, frame_labels = place_clip(
                x,
                rec["onset"],
                rec["offset"],
                rec["kind"],
                frontend,
                cfg,
                rng,
                train,
                path=rec["path"],
                slot=tf.constant(slot, tf.int32),
                noise_bank=noise_bank,
            )
            return feats, wake, frame_labels

        return _map

    def _make(slot: int) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE if train else 1)
        if train:
            # The shards are written grouped by kind (every keyword clip, then every
            # fuzzy negative, then the Speech Commands clips), and a shuffle buffer
            # smaller than a class block produces batches that are almost single
            # class.  That is fatal here: batch-normalisation statistics become
            # class statistics, and the positive-class gradient is applied to whole
            # batches of negatives.  The buffer therefore has to be able to hold the
            # entire split.
            ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True)
        return ds.map(
            _map_factory(slot),
            num_parallel_calls=tf.data.AUTOTUNE if train else 1,
            deterministic=not train,
        )

    if train:
        ds = _make(0)
    else:
        # Dataset.concatenate takes exactly two inputs, so fold left.
        ds = _make(0)
        for i in range(1, EVAL_SLOTS):
            ds = ds.concatenate(_make(i))

    ds = ds.map(
        lambda f, w, fl: (f, {"wake": tf.reshape(w, [-1, 1]), "frame": fl}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds.prefetch(tf.data.AUTOTUNE)


def expand_cv(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Flatten channel dim: ``(B, 49, 40)`` -> ``(B, 49, 40, 1)`` for Conv2D."""
    return ds.map(lambda f, y: (tf.expand_dims(f, -1), y), num_parallel_calls=tf.data.AUTOTUNE)
