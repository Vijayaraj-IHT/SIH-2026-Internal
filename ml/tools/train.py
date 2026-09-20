#!/usr/bin/env python3
"""Train the DS-CNN wake-word detector.

    python -m ml.tools.train --run-name hb-v1
    python -m ml.tools.train --manifest ml/data/manifests/gsc_marvin.json --run-name marvin-v1

Outputs (all under ``artifacts/<run-name>/``):
    model.keras        full multitask model (window head + auxiliary frame head)
    window_head.keras  the single-head model that ships to the device
    history.json       per-epoch losses/metrics
    metrics.json       validation metrics + the selected decision threshold
    config.json        architecture/training hyper-parameters (reproducibility)
    frontend.json      exact front-end constants used for the features
    model_card.md      human-readable card for the SIH judges
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.features import FrontendParams  # noqa: E402
from ml.data.pipeline import (  # noqa: E402
    EVAL_SLOTS,
    AugmentConfig,
    TFFrontend,
    build_dataset,
    expand_cv,
    load_noise_bank,
)
from ml.models.dscnn import (  # noqa: E402
    DSCNNConfig,
    build_dscnn,
    build_multitask_dscnn,
    count_parameters,
    describe,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency, handles ties correctly)."""
    y_true = np.asarray(y_true).astype(np.float64).ravel()
    y_score = np.asarray(y_score).astype(np.float64).ravel()
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank for ties
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precision_recall_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve, stepwise)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y_true.sum()), 1)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev_recall, 0.0)
        prev_recall = max(prev_recall, r)
    return float(ap)


def threshold_at_fa_rate(
    y_true: np.ndarray, y_score: np.ndarray, neg_per_hour: float, clip_hours: float
) -> tuple[float, dict]:
    """Pick tau so that the false-activation rate is at most ``neg_per_hour``.

    The budget is expressed as false activations *per hour of negative audio*,
    not as a raw count: a window-based detector sees ~3670 windows per hour
    (0.98 s windows), so "0.5 FA/hour" is the industry figure of merit, while
    "2% of negative windows" is a completely different (and much easier) bar.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    allow = max(1.0, neg_per_hour * clip_hours)
    neg_scores = np.sort(y_score[y_true == 0])[::-1]
    pos_scores = y_score[y_true == 1]
    if neg_scores.size == 0:
        return 0.5, {"tau": 0.5, "recall": float("nan"), "fa_per_hour": 0.0, "allowed_fa": 0.0}
    if allow >= neg_scores.size:
        tau = float(neg_scores.min()) - 1e-6
    else:
        # tau just above the allow-th largest negative score
        tau = float(neg_scores[int(math.ceil(allow)) - 1]) + 1e-6
    fa = float((neg_scores > tau).sum())
    recall = float((pos_scores > tau).mean()) if pos_scores.size else float("nan")
    return tau, {
        "tau": tau,
        "recall": recall,
        "fa_per_hour": fa / clip_hours if clip_hours > 0 else 0.0,
        "allowed_fa": allow,
        "negatives": int(neg_scores.size),
        "positives": int(pos_scores.size),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def lr_schedule(step: int, total: int, base_lr: float, warmup_frac: float = 0.05) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def masked_bce(y_true: tf.Tensor, y_pred: tf.Tensor, pos_weight: float) -> tf.Tensor:
    """Binary cross-entropy with a positive-class weight.

    ``y_pred`` may carry a trailing singleton dim (the frame head emits
    ``(B, 7, 1)``); it is squeezed so the shapes line up with the scalar-per-step
    targets.
    """
    eps = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.squeeze(tf.cast(y_pred, tf.float32), axis=-1)
    if y_true.shape.rank == y_pred.shape.rank + 1:
        y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    loss = -(pos_weight * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(loss)


def strip_to_window_head(mt_model: tf.keras.Model, cfg: DSCNNConfig) -> tf.keras.Model:
    """Copy the deployed head's weights into a single-output model."""
    single = build_dscnn(cfg)
    for layer in single.layers:
        try:
            src = mt_model.get_layer(layer.name)
        except ValueError:
            continue
        if src.get_weights():
            layer.set_weights(src.get_weights())
    return single


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("ml/data/manifests/hi_bixby.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--run-name", default="hb-v1")
    ap.add_argument("--out-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--width", type=float, default=1.0, help="channel multiplier (0.25 -> tiny model)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--frame-loss-weight", type=float, default=0.3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--head", default="multi", choices=["multi", "single"],
                    help="multi = window head + auxiliary frame head; single = window head only")
    ap.add_argument("--eval-every", type=int, default=1, help="run validation every N epochs")
    ap.add_argument("--augment", default="full", choices=["full", "light", "none"],
                    help="waveform augmentation strength during training")
    ap.add_argument("--placement", default="random", choices=["random", "eval"],
                    help="random = re-sample the offset every epoch (augmentation); "
                         "eval = use the deterministic evaluation placements")
    ap.add_argument("--train-probe", type=int, default=0,
                    help="each epoch, also score N clips of the TRAIN split with eval placement "
                         "(shows whether the model is learning at all, or only memorising)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--target-fa-per-hour", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 epochs, small batches")
    args = ap.parse_args(argv)

    if args.quick:
        args.epochs = 2
        args.batch_size = 64

    tf.keras.utils.set_random_seed(args.seed)
    if args.threads:
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)

    out_dir = args.out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data -----------------------------------------------------------
    cache_index = json.loads((args.cache_dir / "cache_index.json").read_text())
    splits = cache_index["splits"]
    if "train" not in splits or "val" not in splits:
        print(f"ERROR: cache at {args.cache_dir} has no train/val shards", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    frontend = TFFrontend(FrontendParams())
    # The strength of the waveform augmentation is the single most consequential
    # training knob in this project: at 0 dB SNR a large share of the positives
    # are masked beyond recognition, and a model that spends its capacity fitting
    # those windows learns features that do not survive on clean audio.
    aug = {
        "full": AugmentConfig(),
        "light": AugmentConfig(gain_db=3.0, noise_prob=0.5, noise_snr_db=(10.0, 25.0)),
        "none": AugmentConfig(gain_db=0.0, noise_prob=0.0, partial_negative_prob=0.0,
                              silence_prob=0.0, background_prob=0.0),
    }[args.augment]
    print(f"[train] augmentation={args.augment} placement={args.placement}", flush=True)
    noise_bank = load_noise_bank(args.cache_dir)

    ds_train = build_dataset(
        splits["train"]["files"], frontend, aug, args.batch_size, noise_bank, seed=args.seed,
        train=(args.placement == "random"),
    )
    ds_val = build_dataset(
        splits["val"]["files"], frontend, aug, args.batch_size, noise_bank, seed=args.seed, train=False
    )
    ds_train_cv = expand_cv(ds_train)
    ds_val_cv = expand_cv(ds_val)

    # ---- model ----------------------------------------------------------
    cfg = DSCNNConfig(width=args.width, dropout=args.dropout)
    model = build_dscnn(cfg) if args.head == "single" else build_multitask_dscnn(cfg)
    use_frame_head = args.head == "multi" and args.frame_loss_weight > 0
    n_params = count_parameters(model)
    steps_per_epoch = max(1, splits["train"]["clips"] // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] run={args.run_name} params={n_params:,} steps/epoch={steps_per_epoch} total={total_steps}")
    print(f"[train] train clips={splits['train']['clips']} val clips={splits['val']['clips']}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0)
    ckpt_path = out_dir / "best.weights.h5"
    best_ap = -1.0
    best_epoch = 0
    history: list[dict] = []
    step_counter = {"n": 0}
    t_start = time.time()

    @tf.function
    def train_step(feats, targets):
        with tf.GradientTape() as tape:
            preds = model(feats, training=True)
            wake_pred = preds["wake"] if isinstance(preds, dict) else preds
            loss = masked_bce(targets["wake"], wake_pred, args.pos_weight)
            if use_frame_head:
                loss += args.frame_loss_weight * masked_bce(targets["frame"], preds["frame"], 1.0)
            if model.losses:
                loss += tf.add_n(model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss

    # Optional deterministic train-split probe: the same windows every epoch, so
    # train-vs-val trajectories are directly comparable.
    ds_probe = None
    if args.train_probe > 0:
        # ``files`` here are shard paths, so cap the probe in *batches*: each
        # eval batch holds ~batch_size distinct clips at 4 slots each.
        ds_probe = build_dataset(splits["train"]["files"], frontend, aug, args.batch_size, noise_bank,
                                 seed=99, train=False, drop_remainder=False)
        probe_batches = max(1, args.train_probe // args.batch_size)
        ds_probe = ds_probe.take(probe_batches)
        print(f"[train] diagnostic probe: {probe_batches} batches "
              f"(~{probe_batches * args.batch_size * 4} windows, eval placement)", flush=True)

    @tf.function
    def eval_step(feats):
        out = model(feats, training=False)
        return out["wake"] if isinstance(out, dict) else out

    for epoch in range(1, args.epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        for feats, targets in ds_train_cv:
            lr = lr_schedule(step_counter["n"], total_steps, args.lr)
            optimizer.learning_rate.assign(lr)
            loss, loss_wake = train_step(feats, targets)
            ep_loss += float(loss)
            n_batches += 1
            step_counter["n"] += 1

        # ---- validation -------------------------------------------------
        if args.eval_every > 1 and epoch % args.eval_every != 0 and epoch != args.epochs:
            print(f"[epoch {epoch:3d}] loss={ep_loss / max(n_batches, 1):.4f} (validation skipped)", flush=True)
            continue
        probe_auc = probe_ap = float("nan")
        if ds_probe is not None:
            py, ps = [], []
            for feats, targets in ds_probe:
                ps.append(np.asarray(eval_step(feats)).ravel())
                py.append(np.asarray(targets["wake"]).ravel())
            py = np.concatenate(py)
            ps = np.concatenate(ps)
            probe_auc = roc_auc(py, ps)
            probe_ap = precision_recall_auc(py, ps)

        y_true, y_score = [], []
        for feats, targets in ds_val_cv:
            y_score.append(np.asarray(eval_step(feats)).ravel())
            y_true.append(np.asarray(targets["wake"]).ravel())
        y_true = np.concatenate(y_true) if y_true else np.array([])
        y_score = np.concatenate(y_score) if y_score else np.array([])
        auc = roc_auc(y_true, y_score)
        ap_score = precision_recall_auc(y_true, y_score)
        # Window count -> hours: each window is 0.98 s of audio, and the negative
        # windows in the val split are what the FA budget is measured against.
        neg_hours = float((y_true == 0).sum()) * 0.98 / 3600.0
        tau, op = threshold_at_fa_rate(y_true, y_score, args.target_fa_per_hour, max(neg_hours, 1e-6))

        rec = {
            "epoch": epoch,
            "loss": ep_loss / max(n_batches, 1),
            "lr": float(optimizer.learning_rate.numpy()),
            "val_auc": auc,
            "val_ap": ap_score,
            "val_tau": tau,
            "val_recall_at_fa": op["recall"],
            "val_fa_per_hour_at_tau": op["fa_per_hour"],
            "train_probe_auc": probe_auc,
            "train_probe_ap": probe_ap,
        }
        history.append(rec)
        probe_txt = "" if math.isnan(probe_auc) else f" train_auc={probe_auc:.5f} train_ap={probe_ap:.5f}"
        print(
            f"[epoch {epoch:3d}] loss={rec['loss']:.4f} val_auc={auc:.5f} val_ap={ap_score:.5f} "
            f"recall@tau={op['recall']:.4f} (tau={tau:.4f}){probe_txt}",
            flush=True,
        )

        if ap_score > best_ap and not math.isnan(ap_score):
            best_ap = ap_score
            model.save_weights(str(ckpt_path))
            best_epoch = epoch

    if best_ap > 0:
        model.load_weights(str(ckpt_path))

    single = strip_to_window_head(model, cfg) if args.head == "multi" else model
    model.save(str(out_dir / "model.keras"))
    single.save(str(out_dir / "window_head.keras"))

    # ---- final validation with the deployed (single-head) model ---------
    y_true, y_score = [], []
    for feats, targets in ds_val_cv:
        y_score.append(np.asarray(single(feats, training=False)).ravel())
        y_true.append(np.asarray(targets["wake"]).ravel())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    neg_hours = float((y_true == 0).sum()) * 0.98 / 3600.0
    tau, op = threshold_at_fa_rate(y_true, y_score, args.target_fa_per_hour, max(neg_hours, 1e-6))
    metrics = {
        "run_name": args.run_name,
        "keyword": manifest.get("keyword"),
        "best_epoch": best_epoch,
        "val_auc": roc_auc(y_true, y_score),
        "val_ap": precision_recall_auc(y_true, y_score),
        "threshold": tau,
        "target_fa_per_hour": args.target_fa_per_hour,
        "val_recall_at_threshold": op["recall"],
        "val_fa_per_hour": op["fa_per_hour"],
        "val_positive_windows": op["positives"],
        "val_negative_windows": op["negatives"],
        "val_negative_hours": neg_hours,
        "params_total": n_params,
        "train_minutes": (time.time() - t_start) / 60.0,
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "dscnn": cfg.__dict__,
                "augment": aug.__dict__,
                "lr": args.lr,
                "pos_weight": args.pos_weight,
                "frame_loss_weight": args.frame_loss_weight,
                "seed": args.seed,
                "width": args.width,
            },
            indent=2,
        )
    )
    (out_dir / "frontend.json").write_text(json.dumps(frontend.params.to_dict(), indent=2))
    (out_dir / "architecture.json").write_text(json.dumps(describe(model), indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nwrote artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
