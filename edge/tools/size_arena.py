#!/usr/bin/env python3
"""Estimate the TFLite-Micro tensor arena the model needs.

    python -m edge.tools.size_arena --run-name hb-dscnn-w100

TFLite Micro allocates every intermediate tensor out of one contiguous arena.  If
the arena is too small, ``AllocateTensors()`` fails at boot on the device - the
least useful possible failure, since it happens on hardware you may not have in
front of you.  This tool reads the exported ``.tflite`` flatbuffer and simulates
TFLite Micro's allocation:

* every tensor's byte size comes from its shape and dtype,
* input/output/constant tensors do **not** consume arena (weights are read from
  flash, I/O tensors are written by the caller),
* intermediate tensors are laid out with a simple greedy "first tensor whose last
  use has passed wins" strategy, which is what a linear-memory planner does,

and it prints a recommended arena with a safety margin.  The result is a
*planning* estimate: confirm it on the device (the firmware logs
``tensor arena used``) and use the measured value in the write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DTYPE_BYTES = {0: 4, 1: 4, 2: 4, 3: 1, 4: 1, 5: 1, 6: 1, 7: 8, 8: 1, 9: 2, 10: 1}  # TFLite TensorType enum


def analyse(tflite_path: Path) -> dict:
    """Parse the flatbuffer with the TFLite interpreter's own metadata."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    details = interp.get_tensor_details()
    ops = interp._get_ops_details()  # noqa: SLF001

    tensors = []
    for d in details:
        shape = [int(s) for s in d["shape"]]
        dtype = np.dtype(d["dtype"])
        size = int(np.prod(shape) if shape else 1) * dtype.itemsize
        tensors.append({"index": int(d["index"]), "name": d["name"], "shape": shape,
                        "dtype": str(dtype), "bytes": size})

    # first/last use per tensor, from the op list
    first_use: dict[int, int] = {}
    last_use: dict[int, int] = {}
    for step, op in enumerate(ops):
        for t in list(op["inputs"]) + list(op["outputs"]):
            t = int(t)
            if t < 0:
                continue
            first_use.setdefault(t, step)
            last_use[t] = step

    io_indices = {int(d["index"]) for d in interp.get_input_details() + interp.get_output_details()}

    # constants (weights) are read from flash in TFLite Micro - no arena
    weights = []
    intermediates = []
    for t in tensors:
        if t["index"] in io_indices:
            continue
        if t["index"] not in first_use:
            weights.append(t)  # never touched by an op => constant
        else:
            intermediates.append(t)

    # greedy linear-memory planning over intermediate tensors
    live: list[tuple[int, int, int]] = []  # (offset, size, last_use)
    arena_high_water = 0
    plan = []
    for step in range(len(ops) + 1):
        live = [b for b in live if b[2] >= step]
        for t in sorted(intermediates, key=lambda x: x["bytes"], reverse=True):
            if first_use.get(t["index"]) != step:
                continue
            size = t["bytes"]
            used = sorted(live)
            offset = 0
            for off, sz, _last in used:
                if offset + size <= off:
                    break
                offset = max(offset, off + sz)
            live.append((offset, size, last_use[t["index"]]))
            arena_high_water = max(arena_high_water, offset + size)
            plan.append((t["name"], size, offset))
            break  # one new tensor per step keeps the simulation conservative

    total_intermediate_bytes = sum(t["bytes"] for t in intermediates)
    return {
        "model": str(tflite_path),
        "model_bytes": tflite_path.stat().st_size,
        "tensors": len(tensors),
        "constant_bytes": sum(t["bytes"] for t in weights),
        "intermediate_bytes_if_no_reuse": total_intermediate_bytes,
        "arena_estimate_bytes": arena_high_water,
        "largest_intermediate_bytes": max((t["bytes"] for t in intermediates), default=0),
        "io_bytes": sum(t["bytes"] for t in tensors if t["index"] in io_indices),
        "n_ops": len(ops),
        "op_names": sorted({op["op_name"] for op in ops}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="hb-dscnn-w100")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--model", type=Path, default=None, help="explicit .tflite path")
    ap.add_argument("--margin", type=float, default=1.35, help="safety factor applied to the estimate")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    model_path = args.model or (args.artifacts_root / args.run_name / "model_int8.tflite")
    if not model_path.exists():
        print(f"ERROR: {model_path} not found - run `make model-export`", file=sys.stderr)
        return 2

    info = analyse(model_path)
    recommended = int(info["arena_estimate_bytes"] * args.margin)

    print(f"model              : {model_path} ({info['model_bytes'] / 1024:.1f} KiB, {info['n_ops']} ops)")
    print(f"ops                : {', '.join(info['op_names'])}")
    print(f"constant tensors   : {info['constant_bytes'] / 1024:.1f} KiB (read from flash, not arena)")
    print(f"io tensors         : {info['io_bytes']} B")
    print(f"largest intermediate: {info['largest_intermediate_bytes'] / 1024:.1f} KiB")
    print(f"intermediates      : {info['intermediate_bytes_if_no_reuse'] / 1024:.1f} KiB without reuse")
    print(f"arena estimate     : {info['arena_estimate_bytes'] / 1024:.1f} KiB")
    print(f"recommended arena  : {recommended / 1024:.1f} KiB  ({args.margin:.2f}x margin)")
    print()
    print("Set KWS_TENSOR_ARENA_BYTES in edge/esp32/main/kws_app.c to the recommended value,")
    print("then confirm on hardware: the firmware logs 'tensor arena used: N B of M B'.")

    if args.json_out:
        args.json_out.write_text(json.dumps({**info, "recommended_arena_bytes": recommended}, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
