"""The log-mel front-end must be identical on every path that computes it.

Three implementations exist on purpose:

1. ``ml/common/features.py`` - the numpy reference, and the definition of record,
2. the TF graph in ``ml/data/pipeline.py`` - used to train, because it has to run
   inside ``tf.data``,
3. ``edge/common/kws_frontend.c`` - the firmware, which must reproduce the
   training-time features or the exported model is being fed different data.

An inconsistency between any pair of these is the classic silent killer of an
embedded ML project: nothing errors, the model just performs worse than reported.
These tests turn that into a red build.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.common.audio import FRAME_HOP, FRAME_LENGTH, SAMPLE_RATE
from ml.common.features import (
    CONTEXT_FRAMES,
    FFT_SIZE,
    NUM_MEL_BINS,
    FrontendParams,
    hann_periodic,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_to_hz,
)
from ml.data.pipeline import TFFrontend, WINDOW_SAMPLES, build_dataset


def _speechlike(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic signal with speech-like structure (not white noise)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    x = np.zeros(n, dtype=np.float32)
    for f, a in ((120, 1.0), (240, 0.5), (700, 0.3), (1800, 0.15), (3200, 0.08)):
        x += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    # syllable-rate amplitude modulation + a little noise
    x *= 1.0 + 0.7 * np.sin(2 * np.pi * 3.5 * t)
    x += 0.02 * rng.standard_normal(n)
    return (x / np.max(np.abs(x)) * 0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Mel scale / filterbank sanity
# ---------------------------------------------------------------------------
def test_mel_scale_roundtrip() -> None:
    hz = np.array([0.0, 20.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, rtol=1e-9)


def test_mel_filterbank_shape_and_coverage() -> None:
    fb = mel_filterbank()
    assert fb.shape == (NUM_MEL_BINS, FFT_SIZE // 2 + 1)
    assert np.all(fb >= 0.0)
    # every filter must have energy, and the bank must be roughly monotone in
    # centre frequency (a filterbank with a dead or out-of-order band is a classic
    # copy-paste bug)
    centres = [int(np.argmax(row)) for row in fb]
    assert centres == sorted(centres)
    assert all(fb[i].sum() > 0 for i in range(NUM_MEL_BINS))
    assert centres[0] < 20 and centres[-1] > 200


def test_hann_window_is_periodic() -> None:
    """Periodic Hann (w[0] == 0, w[n] == 1 - w[n-N/2]) - not the symmetric one.

    Using ``np.hanning`` here instead of the periodic definition shifts every
    spectrum slightly and is exactly the kind of drift these tests exist to catch.
    """
    w = hann_periodic(FRAME_LENGTH)
    assert w[0] == pytest.approx(0.0, abs=1e-9)
    assert w[FRAME_LENGTH // 2] == pytest.approx(1.0, abs=1e-6)
    assert not np.allclose(w, np.hanning(FRAME_LENGTH))


# ---------------------------------------------------------------------------
# Reference vs TF pipeline
# ---------------------------------------------------------------------------
def test_tf_frontend_matches_numpy_reference() -> None:
    params = FrontendParams()
    tf_fe = TFFrontend(params)
    import tensorflow as tf

    rng = np.random.default_rng(3)
    batch = np.stack([_speechlike(WINDOW_SAMPLES, seed=i) for i in range(3)])

    ref = np.stack([log_mel_spectrogram(x, params, frame_count=CONTEXT_FRAMES) for x in batch])
    got = tf_fe.pad_to_context(tf_fe.batch_logmel(tf.constant(batch))).numpy()

    assert got.shape == ref.shape == (3, CONTEXT_FRAMES, NUM_MEL_BINS)
    # float32 FFT ordering differences are tiny relative to the int8 step (~0.1)
    assert np.max(np.abs(got - ref)) < 1e-3, f"max abs diff {np.max(np.abs(got - ref))}"


def test_padding_before_first_frame_matches() -> None:
    """A short clip must be left-padded with the log floor, not with zeros."""
    params = FrontendParams()
    x = _speechlike(FRAME_LENGTH + 2 * FRAME_HOP)  # only 3 frames of audio
    ref = log_mel_spectrogram(x, params, frame_count=CONTEXT_FRAMES)
    floor = np.log(params.log_floor)
    assert np.allclose(ref[: CONTEXT_FRAMES - 3], floor)
    assert ref[-1].max() > floor


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------
def _features(n: int = 4) -> np.ndarray:
    params = FrontendParams()
    return np.stack(
        [log_mel_spectrogram(_speechlike(WINDOW_SAMPLES, seed=i), params, CONTEXT_FRAMES) for i in range(n)]
    )


def test_quantisation_roundtrip_error_is_within_half_a_step() -> None:
    """Without range clipping the affine map must round-trip to half a step.

    ``tight=False`` uses the full observed range, so every value is representable
    and the only error is the rounding of ``x / scale``.
    """
    from ml.common.features import calibrate_quant_params

    feats = _features()
    q = calibrate_quant_params(feats, tight=False)
    quantised = q.quantize(feats)
    assert quantised.dtype == np.int8
    recovered = q.dequantize(quantised)
    assert np.max(np.abs(recovered - feats)) <= q.scale * 0.51


def test_tight_calibration_only_clips_the_tails() -> None:
    """``tight=True`` buys resolution by clipping; it may only cost the extremes.

    This is the trade the export makes: spending int8 levels on the near-silent
    tail of the log-mel distribution would waste most of the scale on values the
    model does not use, so 0.1/99.9-percentile clipping is deliberate.  What must
    not happen is clipping a meaningful share of the values.
    """
    from ml.common.features import calibrate_quant_params

    feats = _features(8)
    q = calibrate_quant_params(feats)
    recovered = q.dequantize(q.quantize(feats))
    err = np.abs(recovered - feats)
    over = float((err > q.scale * 0.51).mean())
    assert over <= 0.01, f"{over:.2%} of values exceed half a step - calibration is clipping too much"
    # The monotone ordering of mel energies must survive quantisation.
    flat = feats.ravel()
    order = np.argsort(flat)
    assert np.all(np.diff(q.quantize(feats).ravel()[order]) >= 0)


def test_calibration_clips_the_silent_tail() -> None:
    """Tight calibration must not waste int8 levels on near-silent frames."""
    from ml.common.features import calibrate_quant_params

    feats = np.concatenate(
        [
            np.random.default_rng(0).normal(-12.0, 0.4, (100, NUM_MEL_BINS)),  # silence
            np.random.default_rng(1).normal(0.5, 1.5, (100, NUM_MEL_BINS)),  # speech
        ]
    ).astype(np.float32)
    tight = calibrate_quant_params(feats, tight=True)
    loose = calibrate_quant_params(feats, tight=False)
    assert tight.scale < loose.scale
    assert -128 <= tight.zero_point <= 127


# ---------------------------------------------------------------------------
# Front-end geometry
# ---------------------------------------------------------------------------
def test_window_sample_count_produces_exactly_49_frames() -> None:
    from ml.common.audio import n_frames

    params = FrontendParams()
    assert WINDOW_SAMPLES == params.window_samples
    assert n_frames(WINDOW_SAMPLES, FRAME_LENGTH, FRAME_HOP) == CONTEXT_FRAMES
    assert n_frames(WINDOW_SAMPLES - 1, FRAME_LENGTH, FRAME_HOP) == CONTEXT_FRAMES - 1


# ---------------------------------------------------------------------------
# C front-end parity (requires the generated tables + a built host simulator)
# ---------------------------------------------------------------------------
def test_c_frontend_matches_python(tmp_path, repo_root) -> None:
    """Compile the firmware front-end and diff its int8 output against numpy.

    Skipped when the tables header or a C compiler is missing; run
    ``make tables && make -C edge/host_sim`` to enable it.  This is the test that
    lets us claim "the device computes the same features we trained on".
    """
    import shutil
    import subprocess

    tables = repo_root / "edge" / "common" / "kws_frontend_tables.h"
    if not tables.exists():
        pytest.skip("edge/common/kws_frontend_tables.h missing - run `make tables`")
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available")

    sim_dir = repo_root / "edge" / "host_sim"
    build = tmp_path / "kws_host_sim"
    srcs = [
        str(repo_root / "edge" / "common" / "kws_frontend.c"),
        str(repo_root / "edge" / "common" / "kws_g711.c"),
        str(sim_dir / "test_frontend.c"),
    ]
    proc = subprocess.run(
        [cc, "-O2", "-std=c99", "-o", str(build), *srcs, "-lm"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"C build failed:\n{proc.stderr}"

    params = FrontendParams()
    import json as _json

    # the tables header carries the quantisation constants; reload them so the
    # comparison uses the same int8 mapping as the firmware
    header = tables.read_text()
    scale = float(header.split("KWS_QUANT_SCALE ")[1].split("f")[0])
    zero = int(header.split("KWS_QUANT_ZERO_POINT ")[1].split("\n")[0])
    from ml.common.features import QuantParams

    params.quant = QuantParams(scale=scale, zero_point=zero)

    rng = np.random.default_rng(7)
    failures = []
    for i in range(3):
        x = _speechlike(WINDOW_SAMPLES + 1234, seed=100 + i)
        pcm = np.round(np.clip(x, -1, 1 - 2**-15) * 32768).astype("<i2")
        pcm_path = tmp_path / f"in{i}.pcm"
        out_path = tmp_path / f"out{i}.bin"
        pcm_path.write_bytes(pcm.tobytes())
        run = subprocess.run(
            [str(build), "--features", str(pcm_path), str(out_path), str(pcm.size)],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        c_feats = np.frombuffer(out_path.read_bytes(), dtype=np.int8).reshape(CONTEXT_FRAMES, NUM_MEL_BINS)

        ref_float = log_mel_spectrogram(x, params, frame_count=CONTEXT_FRAMES)
        ref_q = params.quant.quantize(ref_float)
        diff = np.abs(c_feats.astype(np.int32) - ref_q.astype(np.int32))
        if diff.max() > 1:
            failures.append((i, int(diff.max()), float(diff.mean())))
    assert not failures, f"C front-end diverges from numpy reference: {failures}"
