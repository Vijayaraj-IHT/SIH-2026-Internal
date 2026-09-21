#!/usr/bin/env python3
"""Evaluate a trained run on the held-out speaker-disjoint test set.

    python -m ml.tools.evaluate --run-name hb-dscnn-w100

What it reports, and why each number is here
--------------------------------------------
The test split contains a speaker that never appeared in training or model
selection (``hb_a3`` for the keyword corpus, plus Speech Commands speakers never
used for training).  So the accuracy here is an honest generalisation estimate.

* **AUC / average precision** - threshold-free discrimination.
* **The precision-recall curve** - because the operating point matters more than
  a single number for a wake word.
* **Recall at a fixed false-activation budget** (FA/hour over held-out negative
  audio) - the metric a product manager actually signs off on.
* **Per-confusable false-activation breakdown** - how often each phonetically
  similar phrase ("Hi Vicky", "Hi Dixie", ...) triggers the detector.  An
  aggregate FA rate hides the fact that one specific phrase is doing most of the
  damage, and that is exactly what a reviewer should want to see.
* **The float vs int8 comparison on the same windows** - the quantisation cost of
  fitting the model into the device budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.features import FrontendParams, QuantParams  # noqa: E402
from ml.common.streaming import WindowScorer, score_waveform_offline  # noqa: E402
from ml.common.synth import load_clips  # noqa: E402
from ml.tools.train import precision_recall_auc, roc_auc, threshold_at_fa_rate  # noqa: E402


def build_eval_windows(cache_dir: Path, frontend, noise_bank, batch: int = 128, split: str = "test"):
    """Score every clip of ``split`` at its 4 deterministic placements.

    Reuses the training pipeline's evaluation mode, so the windows are exactly the
    ones used for model selection - the test set is not redefined downstream.
    """
    from ml.data.pipeline import AugmentConfig, build_dataset

    index = json.loads((cache_dir / "cache_index.json").read_text())
    ds = build_dataset(
        index["splits"][split]["files"],
        frontend,
        AugmentConfig(),
        batch,
        noise_bank,
        seed=0,
        train=False,
        drop_remainder=False,
    )
    return ds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="hb-dscnn-w100")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fa-budgets", default="0.1,0.5,1,5,10", help="false activations per hour to report recall at")
    args = ap.parse_args(argv)

    run_dir = args.artifacts_root / args.run_name
    frontend_file = run_dir / "frontend.json"
    if not frontend_file.exists():
        print(f"ERROR: {run_dir} has no frontend.json - train first", file=sys.stderr)
        return 2

    params = FrontendParams.from_dict(json.loads(frontend_file.read_text()))
    export_report = {}
    if (run_dir / "export_report.json").exists():
        export_report = json.loads((run_dir / "export_report.json").read_text())
        q = export_report.get("feature_quantisation")
        if q:
            params.quant = QuantParams.from_dict(q)

    from ml.data.pipeline import load_noise_bank

    noise_bank = load_noise_bank(args.cache_dir, max_clips=32)

    float_model = tf.keras.models.load_model(run_dir / "window_head.keras")
    float_scorer = WindowScorer(model=float_model)
    int8_scorer = None
    if (run_dir / "model_int8.tflite").exists() and params.quant is not None:
        int8_scorer = WindowScorer(tflite_path=str(run_dir / "model_int8.tflite"), quant=params.quant)

    # ---- score every evaluation window ---------------------------------
    from ml.common.features import FrontendParams as _FP
    from ml.data.pipeline import TFFrontend

    tf_frontend = TFFrontend(params)
    ds = build_eval_windows(args.cache_dir, tf_frontend, noise_bank, args.batch_size, args.split)
    feats, labels = [], []
    for batch_feats, targets in ds:
        feats.append(batch_feats.numpy())
        labels.append(np.asarray(targets["wake"]).ravel())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    float_scores = np.concatenate(
        [float_scorer(feats[i : i + 512]).ravel() for i in range(0, feats.shape[0], 512)]
    )
    int8_scores = None
    if int8_scorer is not None:
        int8_scorer.enable_batching(256)
        int8_scores = np.concatenate(
            [int8_scorer(feats[i : i + 512]).ravel() for i in range(0, feats.shape[0], 512)]
        )

    # ---- metrics --------------------------------------------------------
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    # each window represents one hop of audio; negative "hours" is the amount of
    # non-keyword audio the false-activation budget is spent on
    neg_hours = n_neg * 0.02 / 3600.0

    report: dict = {
        "run_name": args.run_name,
        "split": args.split,
        "windows": int(labels.size),
        "positives": n_pos,
        "negatives": n_neg,
        "negative_hours": neg_hours,
        "float": {
            "auc": roc_auc(labels, float_scores),
            "average_precision": precision_recall_auc(labels, float_scores),
        },
        "operating_points": {},
        "per_confusable": {},
    }

    scorer_used = int8_scores if int8_scores is not None else float_scores
    report["scored_with"] = "tflite_int8" if int8_scores is not None else "keras_float32"
    if int8_scores is not None:
        report["int8"] = {
            "auc": roc_auc(labels, int8_scores),
            "average_precision": precision_recall_auc(labels, int8_scores),
            "mean_abs_score_delta_vs_float": float(np.mean(np.abs(float_scores - int8_scores))),
            "max_abs_score_delta_vs_float": float(np.max(np.abs(float_scores - int8_scores))),
        }

    for budget in [float(x) for x in args.fa_budgets.split(",")]:
        tau, op = threshold_at_fa_rate(labels, scorer_used, budget, neg_hours)
        report["operating_points"][f"{budget}_fa_per_hour"] = {
            "threshold": tau,
            "recall": op["recall"],
            "precision": (op["positives"] * op["recall"]) / max(1e-9, op["positives"] * op["recall"] + op["allowed_fa"]),
            "false_activations_per_hour": op["fa_per_hour"],
        }

    # ---- per-confusable breakdown --------------------------------------
    # Stream confusable clips on their own and count activations per phrase.
    budget = float(args.fa_budgets.split(",")[1] if len(args.fa_budgets.split(",")) > 1 else 0.5)
    tau = report["operating_points"][f"{budget}_fa_per_hour"]["threshold"]
    if args.split == "test":
        from ml.common.synth import build_stream

        rng = np.random.default_rng(4242)
        per_detail: dict[str, dict] = defaultdict(lambda: {"clips": 0, "activations": 0, "hours": 0.0})
        fuzzy = load_clips(args.cache_dir, "test", "fuzzy")
        backgrounds = load_clips(args.cache_dir, "noise_test") + load_clips(args.cache_dir, "noise_bank")
        for detail in sorted({c.detail for c in fuzzy}):
            clips = [c for c in fuzzy if c.detail == detail]
            stream, _events = build_stream(
                keywords=[], negatives=clips, backgrounds=backgrounds, rng=rng,
                duration_s=180.0, include_keywords=False, include_negatives=True,
            )
            scores = score_waveform_offline(stream, scorer_used and (int8_scorer or float_scorer), params)
            # count activations of a debounced detector, not raw threshold crossings
            from ml.common.streaming import PolicyConfig, apply_policy

            dets = apply_policy(scores, PolicyConfig(threshold=tau, confirm_windows=2, refractory_ms=1200))
            entry = per_detail[detail]
            entry["clips"] += len(clips)
            entry["activations"] += len(dets)
            entry["hours"] += stream.size / 16000.0 / 3600.0
        for detail, e in per_detail.items():
            e["activations_per_hour"] = e["activations"] / e["hours"] if e["hours"] else 0.0
        report["per_confusable"] = {k: v for k, v in sorted(per_detail.items())}

    (run_dir / f"evaluation_{args.split}.json").write_text(json.dumps(report, indent=2))

    print(f"=== {args.run_name} on the '{args.split}' split ({report['scored_with']}) ===")
    print(f"windows   : {report['windows']}  ({n_pos} positive / {n_neg} negative)")
    print(f"negative  : {neg_hours * 60:.1f} minutes of held-out non-keyword audio")
    print(f"float     : AUC {report['float']['auc']:.4f}  AP {report['float']['average_precision']:.4f}")
    if "int8" in report:
        print(f"int8      : AUC {report['int8']['auc']:.4f}  AP {report['int8']['average_precision']:.4f}"
              f"  (mean |Δscore| {report['int8']['mean_abs_score_delta_vs_float']:.4f})")
    print("\nrecall at a false-activation budget:")
    for k, v in report["operating_points"].items():
        print(f"  tau={v['threshold']:.4f}  {k:>16s}  recall {100 * v['recall']:.1f}%  precision {100 * v['precision']:.1f}%")
    if report["per_confusable"]:
        print("\nfalse activations per confusable phrase (at "
              f"{budget} FA/h operating point, 3 min each):")
        for detail, e in report["per_confusable"].items():
            flag = "  <-- worst" if e["activations_per_hour"] > 40 else ""
            print(f"  {detail:12s} {e['activations']:3d} activations  {e['activations_per_hour']:6.1f} /hour{flag}")
    print(f"\nwrote {run_dir / f'evaluation_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
