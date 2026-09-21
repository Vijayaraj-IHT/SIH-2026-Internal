#!/usr/bin/env python3
"""End-to-end streaming benchmark: wake-up latency and false activations per hour.

    python -m ml.tools.streaming_eval --run-name hb-dscnn-w100

This is the tool that produces the numbers the SIH evaluation criteria ask for:

* **Latency** - measured from the *annotated end of the keyword* to the instant
  the detector makes the decision (``(k*320 + 480)/16000`` s for window ``k``),
  on continuous audio, over hundreds of real events.
* **False activations** - reported per hour of audio, split by the audio that
  caused them: phonetically confusable words ("Hi Vicky", "Hi Dixie", ...),
  generic speech (Google Speech Commands), and pure background noise.
* **Latency / false-alarm trade-off** - the same streams scored under several
  confirmation policies, because "1 window" is fast but noisy and "3 of 3" is
  slow but calm.  Presenting only one operating point would be dishonest.

Streams are synthesised from held-out speaker data (the corpus's third speaker
plus speaker-disjoint Speech Commands speakers) with real background noise mixed
in at controlled SNR, so every reported number is on audio the model never saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.features import FrontendParams  # noqa: E402
from ml.common.synth import Clip, Event, build_stream, load_clips  # noqa: E402
from ml.common.streaming import (  # noqa: E402
    OnlineWakeDetector,
    PolicyConfig,
    WindowScorer,
    apply_policy,
    detection_time_s,
    score_waveform_offline,
)

# ---------------------------------------------------------------------------
# Scoring / metrics
# ---------------------------------------------------------------------------
def match_detections(
    detections, events: list[Event], tolerance_s: float = 1.5
) -> tuple[list[float], int, int]:
    """Greedy nearest matching between detections and keyword events.

    Returns ``(latencies, matched, false_activations)``.  A detection counts as
    a false activation if no event lies within ``tolerance_s``.
    """
    remaining = list(range(len(events)))
    latencies: list[float] = []
    for det in detections:
        if not remaining:
            break
        best_i, best_d = None, None
        for i in remaining:
            d = abs(events[i].t_end_s - det.time_s)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_d is not None and best_d <= tolerance_s:
            latencies.append(det.time_s - events[best_i].t_end_s)
            remaining.remove(best_i)
    return latencies, len(latencies), len(detections) - len(latencies)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="hb-dscnn-w100")
    ap.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--n-keyword-streams", type=int, default=6)
    ap.add_argument("--n-negative-streams", type=int, default=6)
    ap.add_argument("--n-background-streams", type=int, default=4)
    ap.add_argument("--stream-seconds", type=float, default=240.0)
    ap.add_argument("--background-seconds", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--policies", default="1,2,3", help="confirm_windows values to sweep")
    ap.add_argument("--limit-windows", type=int, default=0, help="debug: cap windows per stream")
    ap.add_argument("--device", default="int8", choices=["int8", "float"], help="scorer to benchmark")
    args = ap.parse_args(argv)

    run_dir = args.artifacts_root / args.run_name
    params = FrontendParams.from_dict(json.loads((run_dir / "frontend.json").read_text()))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    tau = float(metrics["threshold"])
    print(f"[stream] run={args.run_name} threshold={tau:.4f} scorer={args.device}")

    # ---- scorer ---------------------------------------------------------
    scorer_kind = {}
    if args.device == "int8":
        tflite_path = run_dir / "model_int8.tflite"
        if not tflite_path.exists():
            print(f"ERROR: {tflite_path} missing - run ml.tools.export_tflite first", file=sys.stderr)
            return 2
        exp = json.loads((run_dir / "export_report.json").read_text())
        from ml.common.features import QuantParams

        params.quant = QuantParams.from_dict(exp["feature_quantisation"])
        scorer = WindowScorer(tflite_path=str(tflite_path), quant=params.quant)
        if hasattr(scorer, "enable_batching"):
            scorer.enable_batching(args.batch_size)
        scorer_kind = {"backend": "tflite_int8", "batched": True}
    else:
        model = tf.keras.models.load_model(run_dir / "window_head.keras")
        scorer = WindowScorer(model=model)
        scorer_kind = {"backend": "keras_float32"}

    # ---- data -----------------------------------------------------------
    keywords = load_clips(args.cache_dir, "test", "keyword")
    fuzzy = load_clips(args.cache_dir, "test", "fuzzy")
    gsc = load_clips(args.cache_dir, "test", "gsc_word")
    backgrounds = load_clips(args.cache_dir, "noise_test") + load_clips(args.cache_dir, "noise_bank")
    print(f"[stream] clips: keywords={len(keywords)} fuzzy={len(fuzzy)} gsc={len(gsc)} backgrounds={len(backgrounds)}")
    if not keywords:
        print("ERROR: no keyword clips in the test split", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    policies = [int(x) for x in args.policies.split(",")]

    # stream families: (name, kwargs, n_streams)
    families = [
        ("keywords", dict(negatives=gsc, include_keywords=True, include_negatives=True),
         args.n_keyword_streams, args.stream_seconds),
        ("confusables", dict(negatives=fuzzy, include_keywords=False, include_negatives=True),
         args.n_negative_streams, args.stream_seconds),
        ("generic_speech", dict(negatives=gsc, include_keywords=False, include_negatives=True),
         args.n_negative_streams, args.stream_seconds),
        ("background_only", dict(negatives=[], include_keywords=False, include_negatives=False),
         args.n_background_streams, args.background_seconds),
    ]

    report: dict = {
        "run_name": args.run_name,
        "threshold": tau,
        "scorer": scorer_kind,
        "policies": {},
        "streams": [],
        "per_family": {},
    }
    # scores[policy][family] accumulators
    acc: dict = {p: {} for p in policies}
    per_event_rows: list[dict] = []

    for fam_name, kwargs, n_streams, dur in families:
        for s in range(n_streams):
            stream, events = build_stream(
                keywords=keywords,
                negatives=kwargs["negatives"],
                backgrounds=backgrounds,
                rng=rng,
                duration_s=dur,
                include_keywords=kwargs["include_keywords"],
                include_negatives=kwargs["include_negatives"],
            )
            if args.limit_windows:
                stream = stream[: args.limit_windows * 320 + 15840]
                events = [e for e in events if e.t_end_s <= len(stream) / SAMPLE_RATE]
            scores = score_waveform_offline(stream, scorer, params, batch_size=args.batch_size)
            stream_dur = stream.size / SAMPLE_RATE
            stream_report = {
                "family": fam_name,
                "index": s,
                "duration_s": stream_dur,
                "n_events": len(events),
                "n_windows": int(scores.size),
            }
            for p in policies:
                pol = PolicyConfig(threshold=tau, confirm_windows=p, refractory_ms=1200)
                dets = apply_policy(scores, pol)
                if fam_name == "keywords":
                    lat, matched, fa = match_detections(dets, events)
                else:
                    lat, matched, fa = [], 0, len(dets)
                stream_report[f"policy_{p}"] = {
                    "detections": len(dets),
                    "matched": matched,
                    "false_activations": fa,
                    "fa_per_hour": fa / (stream_dur / 3600.0),
                    "median_latency_ms": percentile(lat, 50) * 1000 if lat else None,
                    "p90_latency_ms": percentile(lat, 90) * 1000 if lat else None,
                }
                fam_acc = acc[p].setdefault(
                    fam_name, {"hours": 0.0, "detections": 0, "false_activations": 0, "latencies": [], "events": 0}
                )
                fam_acc["hours"] += stream_dur / 3600.0
                fam_acc["detections"] += len(dets)
                fam_acc["false_activations"] += fa
                fam_acc["events"] += len(events)
                fam_acc["latencies"].extend(lat)
                if fam_name == "keywords" and p == policies[0]:
                    for e in events:
                        per_event_rows.append(
                            {
                                "stream": s,
                                "t_end_s": round(e.t_end_s, 4),
                                "snr_db": e.snr_db,
                                "detail": e.clip.detail,
                                "speaker": e.clip.speaker,
                                "speech_ms": round(e.clip.speech_samples / SAMPLE_RATE * 1000, 1),
                            }
                        )
            report["streams"].append(stream_report)
            print(
                f"[stream] {fam_name}#{s}: {stream_dur:.0f}s events={len(events)} "
                + " ".join(
                    f"p{p}:det={stream_report[f'policy_{p}']['detections']}"
                    f"/fa={stream_report[f'policy_{p}']['fa_per_hour']:.1f}ph"
                    f"/lat={stream_report[f'policy_{p}']['median_latency_ms']}"
                    for p in policies
                ),
                flush=True,
            )

    # ---- aggregate ------------------------------------------------------
    for p in policies:
        fams = acc[p]
        agg = {}
        for fam, d in fams.items():
            lat = d["latencies"]
            agg[fam] = {
                "hours": d["hours"],
                "events": d["events"],
                "detections": d["detections"],
                "false_activations": d["false_activations"],
                "fa_per_hour": d["false_activations"] / d["hours"] if d["hours"] else 0.0,
                "recall": (len(lat) / d["events"]) if d["events"] else None,
                "median_latency_ms": percentile(lat, 50) * 1000 if lat else None,
                "p90_latency_ms": percentile(lat, 90) * 1000 if lat else None,
                "p95_latency_ms": percentile(lat, 95) * 1000 if lat else None,
                "max_latency_ms": max(lat) * 1000 if lat else None,
                "early_detections": sum(1 for x in lat if x < 0) if lat else 0,
            }
        report["per_family"][f"confirm_{p}"] = agg
        report["policies"][f"confirm_{p}"] = {
            "confirm_windows": p,
            "latency_ms": p * 20.0,  # one 20 ms hop per extra confirming window
            "keywords": agg.get("keywords", {}),
            "confusables_fa_per_hour": agg.get("confusables", {}).get("fa_per_hour"),
            "generic_speech_fa_per_hour": agg.get("generic_speech", {}).get("fa_per_hour"),
            "background_fa_per_hour": agg.get("background_only", {}).get("fa_per_hour"),
        }

    # ---- write artifacts ------------------------------------------------
    (run_dir / "streaming_report.json").write_text(json.dumps(report, indent=2))
    if per_event_rows:
        (run_dir / "streaming_events.json").write_text(json.dumps(per_event_rows, indent=2))

    print("\n=== latency / false-alarm trade-off (deployed scorer) ===")
    for p in policies:
        k = report["policies"][f"confirm_{p}"]["keywords"]
        print(
            f"confirm {p}-of-{p}: median latency {k.get('median_latency_ms', float('nan')):.0f} ms "
            f"(p90 {k.get('p90_latency_ms', float('nan')):.0f} ms), "
            f"recall {100 * (k.get('recall') or 0):.1f}%, "
            f"FA/h: confusables {report['policies'][f'confirm_{p}']['confusables_fa_per_hour']:.2f} "
            f"generic {report['policies'][f'confirm_{p}']['generic_speech_fa_per_hour']:.2f} "
            f"background {report['policies'][f'confirm_{p}']['background_fa_per_hour']:.2f}"
        )

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lat = acc[policies[0]]["keywords"]["latencies"]
        if lat:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].hist(np.array(lat) * 1000, bins=30, color="#2b6cb0", edgecolor="white")
            axes[0].set_xlabel("wake-up latency (ms, from end of keyword)")
            axes[0].set_ylabel("events")
            axes[0].set_title(f"Latency distribution ({len(lat)} events, {args.device})")
            axes[0].axvline(np.median(lat) * 1000, color="#c53030", linestyle="--", label=f"median {np.median(lat)*1000:.0f} ms")
            axes[0].legend()
            sorted_lat = np.sort(np.array(lat) * 1000)
            axes[1].plot(sorted_lat, np.arange(1, sorted_lat.size + 1) / sorted_lat.size, color="#2f855a")
            axes[1].set_xlabel("latency (ms)")
            axes[1].set_ylabel("fraction of events ≤ x")
            axes[1].set_title("Latency CDF")
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "streaming_latency.png", dpi=140)
            print(f"[stream] wrote {run_dir / 'streaming_latency.png'}")
    except Exception as exc:  # plotting is a nicety, never a hard failure
        print(f"[stream] plot skipped: {exc}")

    print(f"\nwrote {run_dir / 'streaming_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
