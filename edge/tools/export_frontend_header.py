#!/usr/bin/env python3
"""Generate the C tables header consumed by the firmware front-end.

    python -m edge.tools.export_frontend_header --run-name hb-dscnn-w100

Everything the C code needs that must not drift from training:

* the periodic Hann window,
* the Slaney-normalised mel filterbank,
* the log floor,
* the affine int8 quantisation constants for the feature tensor, and
* the decimation FIR used by the mu-law uplink.

Output: ``edge/common/kws_frontend_tables.h`` (checked into Git so the firmware
builds without running Python).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.features import FrontendParams, hann_periodic, mel_filterbank  # noqa: E402


def c_float_array(name: str, values: np.ndarray, shape: tuple[int, ...], per_line: int = 6, fmt: str = ".8f") -> str:
    """Emit a C float array.

    A 2-D array is emitted with an explicit brace per row.  Writing it as one flat
    list is legal C but triggers ``-Wmissing-braces`` in every translation unit that
    includes the header, and a header that makes the build noisy is a header people
    stop reading.
    """
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.ravel()
    dims = ", ".join(str(s) for s in shape)
    lines = [f"/* {name}: shape {{{dims}}} */".replace("{{", "{").replace("}}", "}")]
    if len(shape) == 1:
        lines.append(f"static const float {name}[{shape[0]}] = {{")
        for i in range(0, len(flat), per_line):
            lines.append("    " + ", ".join(f"{v:{fmt}}f" for v in flat[i : i + per_line]) + ",")
    elif len(shape) == 2:
        lines.append(f"static const float {name}[{shape[0]}][{shape[1]}] = {{")
        for row in arr:
            lines.append("    {")
            for i in range(0, len(row), per_line):
                lines.append("        " + ", ".join(f"{v:{fmt}}f" for v in row[i : i + per_line]) + ",")
            lines.append("    },")
    else:
        raise ValueError(f"c_float_array handles 1-D and 2-D arrays, got shape {shape}")
    lines.append("};")
    return "\n".join(lines)


def design_decimation_fir(taps: int = 15, cutoff: float = 0.25) -> np.ndarray:
    """Windowed-sinc low-pass at ``cutoff`` (normalised to the 16 kHz Nyquist).

    ``cutoff=0.25`` -> 4 kHz, which is the band G.711 at 8 kHz can carry.  A
    Hamming window keeps the stopband low enough that aliasing stays below the
    quantisation noise of the 8-bit codec itself.
    """
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= np.hamming(taps)
    h /= h.sum()
    return h.astype(np.float64)



# ---------------------------------------------------------------------------
# TFLite-Micro operator resolver
# ---------------------------------------------------------------------------
# TFLite op name -> MicroMutableOpResolver method suffix.  Listing the ops
# explicitly (instead of AllOpsResolver) keeps ~30 KB of dead kernels out of the
# firmware, but it means the list must match the graph exactly - so it is
# generated from the exported model's own op inventory rather than hand-written.
# A missing op shows up as a failed AllocateTensors() at boot, which is a
# terrible way to find out.
OP_TO_RESOLVER = {
    "CONV_2D": "Conv2D",
    "DEPTHWISE_CONV_2D": "DepthwiseConv2D",
    "FULLY_CONNECTED": "FullyConnected",
    "MEAN": "Mean",
    "MUL": "Mul",
    "ADD": "Add",
    "SUB": "Sub",
    "LOGISTIC": "Logistic",
    "RELU": "Relu",
    "RELU6": "Relu6",
    "RESHAPE": "Reshape",
    "QUANTIZE": "Quantize",
    "DEQUANTIZE": "Dequantize",
    "SOFTMAX": "Softmax",
    "AVERAGE_POOL_2D": "AveragePool2D",
    "MAX_POOL_2D": "MaxPool2D",
    "PAD": "Pad",
    "STRIDED_SLICE": "StridedSlice",
    "SPLIT": "Split",
    "CONCATENATION": "Concatenation",
    "TRANSPOSE": "Transpose",
    "SQUEEZE": "Squeeze",
    "EXPAND_DIMS": "ExpandDims",
    "SUM": "Sum",
    "REDUCE_MAX": "ReduceMax",
    "TANH": "Tanh",
    "GATHER": "Gather",
}


#: Quantisation used when no trained run supplies its calibrated constants.
#:
#: The window, the mel filterbank and the FIR taps are pure functions of the
#: front-end parameters - they do **not** depend on training, so a fresh clone can
#: build and parity-test the C front-end. Only the int8 affine constants come from
#: the exported model. This pair is a nominal range for log-mel in this front-end
#: (the floor is ln(1e-6) = -13.8, real speech peaks a few dB above 0), and it must
#: be replaced by the calibrated values before flashing a device - the firmware
#: would otherwise dequantise the model's input with the wrong scale. It is fine
#: for the parity test, which only needs the C and Python sides to agree.
NOMINAL_FEATURE_RANGE = (-14.0, 4.0)


def default_quant_params() -> "QuantParams":
    from ml.common.features import QuantParams

    lo, hi = NOMINAL_FEATURE_RANGE
    scale = (hi - lo) / 255.0
    zero_point = int(round(-128.0 - lo / scale))
    return QuantParams(scale=scale, zero_point=max(-128, min(127, zero_point)))


def write_ops_include(export_report: dict, out_path: Path) -> None:
    """Write the ``#include``-able op list consumed by kws_runner.cc."""
    ops = export_report.get("ops") or export_report.get("op_inventory") or {}
    names = sorted(ops.keys() if isinstance(ops, dict) else ops)
    if not names:
        print("[ops] WARNING: no op inventory in export_report.json - skipping", file=sys.stderr)
        return
    unknown = [n for n in names if n not in OP_TO_RESOLVER]
    if unknown:
        print(f"[ops] ERROR: unmapped TFLite ops {unknown} - add them to OP_TO_RESOLVER", file=sys.stderr)
        raise SystemExit(3)

    lines = [
        "/*",
        " * AUTO-GENERATED by edge/tools/export_frontend_header.py - do not edit.",
        " *",
        f" * TFLite-Micro resolver entries required by {export_report.get('run_name', 'the exported model')}.",
        " * Included by edge/esp32/main/kws_runner.cc:",
        " *     #define KWS_ADD_OP(name) resolver.Add##name();",
        " *     #include \"kws_ops.inc\"",
        " *     #undef KWS_ADD_OP",
        " */",
        f"#define KWS_NUM_OPS {len(names)}",
        "",
    ]
    for n in names:
        count = ops[n] if isinstance(ops, dict) else ""
        lines.append(f"KWS_ADD_OP({OP_TO_RESOLVER[n]})  /* {n}: {count} */" if count != "" else f"KWS_ADD_OP({OP_TO_RESOLVER[n]})  /* {n} */")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ops] wrote {out_path} with {len(names)} ops: {', '.join(names)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="hb-dscnn-w100")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("edge/common/kws_frontend_tables.h"))
    ap.add_argument("--fir-taps", type=int, default=15)
    ap.add_argument("--ops-out", type=Path, default=Path("edge/esp32/main/kws_ops.inc"),
                    help="generated operator list for the TFLite-Micro resolver")
    ap.add_argument("--allow-default-quant", action="store_true",
                    help="build the tables even without a calibrated export, using a nominal "
                         "feature range. Enough to compile and parity-test the front-end; NOT "
                         "enough to deploy, because the device would dequantise with the wrong scale.")
    args = ap.parse_args(argv)

    params = FrontendParams()
    run_dir = args.artifacts_root / args.run_name
    frontend_file = run_dir / "frontend.json"
    if frontend_file.exists():
        d = json.loads(frontend_file.read_text())
        params = FrontendParams.from_dict(d)
        # Prefer the export-time quantisation constants (the ones baked into the model).
        export_report = run_dir / "export_report.json"
        if export_report.exists():
            q = json.loads(export_report.read_text()).get("feature_quantisation")
            if q:
                from ml.common.features import QuantParams

                params.quant = QuantParams.from_dict(q)
        print(f"[tables] using front-end from {frontend_file}")
    else:
        print(f"[tables] WARNING: {frontend_file} not found, using defaults (no quantisation!)")

    if params.quant is None:
        if not args.allow_default_quant:
            print(
                "[tables] ERROR: no quantisation parameters available - run ml.tools.export_tflite first, "
                "or pass --allow-default-quant to build uncalibrated tables for a compile/parity check",
                file=sys.stderr,
            )
            return 2
        params.quant = default_quant_params()
        print(
            f"[tables] WARNING: building UNCALIBRATED tables (scale={params.quant.scale:.6f}, "
            f"zero_point={params.quant.zero_point}) - compile/parity only, do NOT flash this",
            file=sys.stderr,
        )

    window = hann_periodic(params.frame_length)
    mel = mel_filterbank(params.sample_rate, params.fft_size, params.num_mel_bins, params.lower_hz, params.upper_hz)
    fir = design_decimation_fir(args.fir_taps)

    header = f"""/*
 * AUTO-GENERATED by edge/tools/export_frontend_header.py - do not edit.
 *
 * Source run      : {args.run_name}
 * Sample rate     : {params.sample_rate} Hz
 * Frame / hop     : {params.frame_length} / {params.frame_hop} samples
 * FFT size        : {params.fft_size}
 * Mel bins        : {params.num_mel_bins} ({params.lower_hz:.0f}-{params.upper_hz:.0f} Hz)
 * Context frames  : {params.context_frames}
 * Feature quant   : scale={params.quant.scale:.8f} zero_point={params.quant.zero_point}
 *
 * These constants are identical to the ones used to train and export the model;
 * tests/test_features.py::test_c_frontend_matches_python fails if the C front-end and
 * reference disagree.
 */
#ifndef KWS_FRONTEND_TABLES_H
#define KWS_FRONTEND_TABLES_H

#define KWS_QUANT_SCALE {params.quant.scale:.8f}f
#define KWS_QUANT_ZERO_POINT {params.quant.zero_point}
#define KWS_LOG_FLOOR {params.log_floor:.8f}f
#define KWS_DECIM_FIR_TAPS {args.fir_taps}

{c_float_array("KWS_HANN_WINDOW", window, (params.frame_length,))}

{c_float_array("KWS_MEL_MATRIX", mel, (params.num_mel_bins, params.fft_size // 2 + 1))}

{c_float_array("KWS_DECIM_FIR", fir, (args.fir_taps,))}

#endif /* KWS_FRONTEND_TABLES_H */
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header)
    print(f"[tables] wrote {args.out} ({len(header) / 1024:.1f} KiB, {args.out.read_text().count(chr(10))} lines)")
    print(f"[tables] quant: scale={params.quant.scale:.8f} zero_point={params.quant.zero_point}")

    report_file = run_dir / "export_report.json"
    if report_file.exists():
        write_ops_include(json.loads(report_file.read_text()), args.ops_out)
    else:
        print("[ops] export_report.json not found - run ml.tools.export_tflite first", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
