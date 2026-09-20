"""Build a TFRecord cache of 16 kHz mono waveforms from the raw corpora.

Two reasons this step exists:

1. **Speed.** The corpora are a mix of 16 kHz mono, 44.1 kHz and 48 kHz stereo
   WAVs.  Doing the decode/resample once, up front, keeps the training input
   pipeline pure-TensorFlow (no ``tf.numpy_function`` stalls) and makes an
   epoch cost seconds instead of minutes.

2. **Voice-activity boundaries.** For every clip we store the first and last
   sample that carry speech energy.  This is what lets the streaming simulator
   splice clips into a continuous stream and then measure the latency from the
   *true* end of the keyword to the *detected* end - without it, the latency
   number would be a guess.

Audio is stored as raw int16 bytes, so a 1 s clip is 32 KB and the whole cache
for ``hi_bixby`` + 12k GSC negatives is well under 1 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.audio import (  # noqa: E402
    FRAME_HOP,
    FRAME_LENGTH,
    SAMPLE_RATE,
    load_audio,
    pcm16_from_float,
)

AUDIO_KEY = "audio"
FEATURE_SPEC = {
    AUDIO_KEY: tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "speaker": tf.io.FixedLenFeature([], tf.string),
    "source": tf.io.FixedLenFeature([], tf.string),
    "kind": tf.io.FixedLenFeature([], tf.string),
    "detail": tf.io.FixedLenFeature([], tf.string),
    "split": tf.io.FixedLenFeature([], tf.string),
    "onset": tf.io.FixedLenFeature([], tf.int64),
    "offset": tf.io.FixedLenFeature([], tf.int64),
    "duration": tf.io.FixedLenFeature([], tf.int64),
    "path": tf.io.FixedLenFeature([], tf.string),
}


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------
def speech_bounds(
    x: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop: int = FRAME_HOP,
    dynamic_range_db: float = 25.0,
    floor_dbfs: float = -55.0,
    max_seconds: float = 1.1,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[int, int]:
    """Return ``(onset, offset)`` sample indices of the utterance.

    This is the single most consequential function in the data pipeline, and the
    first version of it was wrong in a way that made the model untrainable.

    The corpora annotate a *recording*, not a *word*: a positive clip is 1.7-6.4 s
    long (mean 2.8 s) and contains one "hi bixby" (about 0.7 s) somewhere inside,
    sometimes over loud background noise.  Picking the *longest* run of loud frames
    therefore often selects the noise instead of the word, which silently labels a
    window of street noise as a positive example.  With ~40% of the positive class
    mislabelled, the best a model can do is learn the recording conditions.

    So the span is **anchored on the loudest frame** instead of on the longest run:

    * the threshold is relative to the peak frame's level, not to the clip maximum,
    * the span grows outward from the peak and stops where the level drops,
    * and it is capped at ``max_seconds`` (the keyword cannot be longer than this),
      trimming symmetrically around the peak.

    The cap matters: without it, a noisy recording whose noise floor happens to be
    above the relative threshold yields a span of the whole clip, and the window
    placement then puts a positive window nowhere near the word.
    """
    n = x.size
    if n < frame_length:
        return 0, n

    starts = np.arange(0, n - frame_length + 1, hop)
    frames = x[starts[:, None] + np.arange(frame_length)[None, :]]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))

    peak_i = int(np.argmax(rms_db))
    thresh_db = max(floor_dbfs, float(rms_db[peak_i]) - dynamic_range_db)
    active = rms_db > thresh_db

    # grow left/right from the peak while the level stays up
    lo = hi = peak_i
    while lo > 0 and active[lo - 1]:
        lo -= 1
    while hi + 1 < active.size and active[hi + 1]:
        hi += 1

    onset = max(0, int(starts[lo]) - frame_length // 2)
    offset = min(n, int(starts[hi]) + hop + frame_length // 2)

    max_samples = int(max_seconds * sample_rate)
    if offset - onset > max_samples:
        # Keep the loudest ``max_samples`` centred on the peak frame.
        centre = int(starts[peak_i]) + frame_length // 2
        onset = max(0, centre - max_samples // 2)
        offset = min(n, onset + max_samples)
        onset = max(0, offset - max_samples)

    if offset <= onset:
        return 0, n
    return onset, offset


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _example(rec: dict, audio: bytes) -> tf.train.Example:
    def b(v: str) -> tf.train.Feature:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode("utf-8")]))

    def i(v: int) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                AUDIO_KEY: b_literal(audio),
                "label": i(rec["label"]),
                "speaker": b(rec["speaker"]),
                "source": b(rec["source"]),
                "kind": b(rec["kind"]),
                "detail": b(rec["detail"]),
                "split": b(rec["split"]),
                "onset": i(rec["onset"]),
                "offset": i(rec["offset"]),
                "duration": i(rec["duration"]),
                "path": b(rec["path"]),
            }
        )
    )


def b_literal(audio: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio]))


def build_cache(
    manifest_path: Path,
    data_dir: Path,
    out_dir: Path,
    shard_size: int = 2000,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    payload = json.loads(manifest_path.read_text())
    clips = payload["clips"]
    if limit:
        clips = clips[:limit]
    by_split: dict[str, list[dict]] = {}
    for rec in clips:
        by_split.setdefault(rec["split"], []).append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"manifest": str(manifest_path), "keyword": payload.get("keyword"), "splits": {}}

    for split, recs in sorted(by_split.items()):
        shard_idx = 0
        written = 0
        n_speech = 0
        n_empty = 0
        writer = None
        files: list[str] = []
        try:
            for i, rec in enumerate(recs):
                if i % shard_size == 0:
                    if writer is not None:
                        writer.close()
                    shard_idx += 1
                    path = out_dir / f"{split}-{shard_idx:03d}.tfrec"
                    files.append(str(path))
                    writer = tf.io.TFRecordWriter(str(path))
                try:
                    x = load_audio(data_dir / rec["path"], SAMPLE_RATE)
                except Exception as exc:  # pragma: no cover - corrupt file
                    if verbose:
                        print(f"  skip {rec['path']}: {exc}")
                    continue
                if x.size < FRAME_LENGTH:
                    n_empty += 1
                    continue
                onset, offset = speech_bounds(x)
                if float(np.max(np.abs(x))) < 1e-4:
                    n_empty += 1
                    continue
                if split.startswith("noise"):
                    # Background recordings: no speech to locate, keep whole clip.
                    onset, offset = 0, x.size
                else:
                    n_speech += offset > onset
                pcm = pcm16_from_float(x)
                rec = dict(rec)
                rec.update(
                    onset=int(onset),
                    offset=int(offset),
                    duration=int(x.size),
                    path=rec["path"],
                )
                writer.write(_example(rec, pcm.tobytes()).SerializeToString())
                written += 1
                if verbose and written % 2000 == 0:
                    print(f"  {split}: {written} clips", flush=True)
        finally:
            if writer is not None:
                writer.close()
        stats["splits"][split] = {
            "clips": written,
            "shards": len(files),
            "with_speech": n_speech,
            "skipped": n_empty,
            "files": files,
        }
        if verbose:
            print(f"[cache] {split}: {written} clips in {len(files)} shard(s)")

    (out_dir / "cache_index.json").write_text(json.dumps(stats, indent=2))
    return stats


def parse_example(serialized: tf.Tensor) -> dict:
    """TFRecord -> python dict of tensors (audio still raw int16 bytes)."""
    return tf.io.parse_single_example(serialized, FEATURE_SPEC)


def decode_audio_int16(features: dict) -> tf.Tensor:
    return tf.io.decode_raw(features[AUDIO_KEY], tf.int16)


def load_feature_params(cache_dir: Path):
    """The front-end constants used at training time (mirrors FrontendParams)."""
    from ml.common.features import FrontendParams

    idx = cache_dir / "cache_index.json"
    if idx.exists():
        _ = json.loads(idx.read_text())
    return FrontendParams()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "kws-datasets" / "raw")
    ap.add_argument("--out", type=Path, default=Path("data/cache"))
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    stats = build_cache(args.manifest, args.data_dir, args.out, args.shard_size, args.limit)
    print(json.dumps(stats["splits"], indent=2))
    print(f"\nwrote cache index to {args.out / 'cache_index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
