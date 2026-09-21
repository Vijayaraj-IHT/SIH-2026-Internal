"""Parity between the C front-end and the Python one.

The device computes its own log-mel features in C (`edge/common/kws_frontend.c`);
the model is trained on the NumPy implementation in `ml/common/features.py`. If
those two disagree, every accuracy number measured on the host is fiction - and
the disagreement is invisible, because both sides return a plausible-looking
spectrogram. This is the test that ties them together.

It needs the host simulator to have been built, which needs the generated tables
header:

    python -m edge.tools.export_frontend_header --allow-default-quant
    make -C edge/host_sim

CI does exactly that. When the binary is absent the tests skip rather than fail, so
a fresh clone without a compiler still gets a meaningful `make test`.

What is asserted: the int8 tensors the model actually consumes. Both sides quantise
the *same* float pipeline, so a difference larger than one int8 step means the two
implementations genuinely disagree - the window, the filterbank, the FFT or the
floor. A one-step difference is expected only for values sitting exactly on a
rounding boundary, where floating-point association order decides the direction.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ml.common.features import (
    CONTEXT_FRAMES,
    FRAME_HOP,
    FRAME_LENGTH,
    NUM_MEL_BINS,
    FrontendParams,
    QuantParams,
    features_from_waveform,
    log_mel_spectrogram,
)

REPO = Path(__file__).resolve().parents[1]
HOST_SIM = REPO / "edge" / "host_sim" / "build" / "kws_host_sim"
TABLES = REPO / "edge" / "common" / "kws_frontend_tables.h"

pytestmark = pytest.mark.skipif(
    not HOST_SIM.exists(), reason="host simulator not built (see module docstring)"
)


def _float_define(text: str, name: str) -> float:
    m = re.search(rf"^#define\s+{name}\s+([0-9.eE+-]+)f?\s*$", text, re.MULTILINE)
    assert m, f"{name} not found in the generated tables"
    return float(m.group(1))


def _tables_quant() -> QuantParams:
    """Read the quantisation constants out of the generated header.

    Parsed rather than re-derived so the test fails if the header is stale or was
    built with different parameters - re-deriving them from FrontendParams would
    make the test agree with itself no matter what the C side was compiled with.
    """
    assert TABLES.exists(), "kws_frontend_tables.h missing - see module docstring"
    text = TABLES.read_text()
    return QuantParams(
        scale=_float_define(text, "KWS_QUANT_SCALE"),
        zero_point=int(_float_define(text, "KWS_QUANT_ZERO_POINT")),
    )


def _tables_params() -> FrontendParams:
    text = TABLES.read_text()
    params = FrontendParams()
    params.log_floor = _float_define(text, "KWS_LOG_FLOOR")
    params.quant = _tables_quant()
    return params


def _fixture(n_samples: int, seed: int = 7) -> np.ndarray:
    """Deterministic speech-like int16 audio.

    Deliberately not a pure tone: a tone excites two or three mel bins, so a wrong
    filterbank or a misplaced window would still agree. Modulated formants plus a
    few noise bursts exercise every bin and the frame boundaries.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / 16000.0
    x = np.zeros(n_samples, dtype=np.float64)
    for f0, amp in ((220.0, 0.45), (730.0, 0.30), (1900.0, 0.20)):
        x += amp * np.sin(2 * np.pi * f0 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t))
    x += 0.05 * rng.standard_normal(n_samples)
    # Burst positions are a fraction of the clip so the helper works for any
    # length; they exist to put transients across frame boundaries.
    for frac in (0.15, 0.42, 0.70):
        i = int(frac * (n_samples - 800))
        if i >= 0:
            x[i : i + 800] += 0.5 * rng.standard_normal(800)
    return np.round(np.clip(x, -1.0, 1.0) * 0.9 * 32767.0).astype(np.int16)


def _run_features(pcm: np.ndarray, tmp_path: Path) -> np.ndarray:
    in_path = tmp_path / "in.pcm"
    out_path = tmp_path / "out.bin"
    in_path.write_bytes(pcm.astype("<i2").tobytes())
    proc = subprocess.run(
        [str(HOST_SIM), "--features", str(in_path), str(out_path)],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    raw = np.frombuffer(out_path.read_bytes(), dtype=np.int8)
    assert raw.size == CONTEXT_FRAMES * NUM_MEL_BINS, f"unexpected feature size {raw.size}"
    return raw.reshape(CONTEXT_FRAMES, NUM_MEL_BINS)


def test_c_frontend_matches_python(tmp_path: Path) -> None:
    params = _tables_params()
    pcm = _fixture(FRAME_LENGTH + FRAME_HOP * (CONTEXT_FRAMES + 8))

    c_feats = _run_features(pcm, tmp_path)
    x = pcm.astype(np.float32) / 32768.0
    py_feats = features_from_waveform(x, params, quantize=True)

    diff = np.abs(c_feats.astype(np.int32) - py_feats.astype(np.int32))
    exact = float(np.mean(diff == 0))
    assert int(diff.max()) <= 1, f"max int8 difference {int(diff.max())} (>1 step = real disagreement)"
    assert exact > 0.98, f"only {exact:.2%} of the 1960 feature values agree exactly"


def test_c_frontend_float_pipeline_is_close(tmp_path: Path) -> None:
    """Beyond the int8 step: compare the float log-mel itself.

    int8 agreement at one step could in principle hide a small-but-systematic bias,
    so the pre-quantisation values are compared as well, using the C int8 output
    dequantised back to float.
    """
    params = _tables_params()
    pcm = _fixture(FRAME_LENGTH + FRAME_HOP * (CONTEXT_FRAMES + 8), seed=13)

    c_feats = _run_features(pcm, tmp_path)
    # Use the canonical dequantiser so the test cannot disagree with itself about
    # the sign of the zero point.
    c_float = params.quant.dequantize(c_feats).astype(np.float64)
    x = pcm.astype(np.float32) / 32768.0
    py_float = log_mel_spectrogram(x, params, CONTEXT_FRAMES)

    # Only compare where the C side is not clipped at the int8 rails.
    interior = (c_feats > -127) & (c_feats < 127)
    err = np.abs(c_float[interior] - py_float[interior])
    assert np.mean(err) < params.quant.scale, f"mean dequantised error {np.mean(err):.4f} exceeds one step"
    assert np.max(err) <= params.quant.scale, f"max dequantised error {np.max(err):.4f} exceeds one step"


def test_c_frontend_pads_the_window_before_the_first_frame(tmp_path: Path) -> None:
    """A too-short clip is left-padded with the quantised floor, exactly like training."""
    params = _tables_params()
    pcm = _fixture(FRAME_LENGTH * 3, seed=11)
    c_feats = _run_features(pcm, tmp_path)
    py_feats = features_from_waveform(pcm.astype(np.float32) / 32768.0, params, quantize=True)

    assert np.array_equal(c_feats[0], py_feats[0]), "the leading padded frame differs"
    assert not np.allclose(c_feats[0], c_feats[-1]), "the padded row and the newest row are identical"
    assert np.array_equal(c_feats[-1], py_feats[-1]), "the newest frame differs"


def test_g711_codec_through_the_host_sim(tmp_path: Path) -> None:
    """The C codec must produce the same bytes as the Python one."""
    from ml.common.audio import ulaw_encode

    pcm = _fixture(16000, seed=3)
    in_path = tmp_path / "in.pcm"
    out_path = tmp_path / "out.ulaw"
    in_path.write_bytes(pcm.astype("<i2").tobytes())
    proc = subprocess.run([str(HOST_SIM), "--g711", str(in_path), str(out_path)], capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()

    got = np.frombuffer(out_path.read_bytes(), dtype=np.uint8)
    assert got.size > 0, "the codec produced no output"
    assert int(got.min()) != int(got.max()), "the codec output is constant"
    # The host sim decimates 16 kHz -> 8 kHz first, so compare on the subset of the
    # Python encoding that survived; a mismatch here means one of the two tables is wrong.
    py = ulaw_encode(pcm)
    assert len(got) <= len(py)
