"""Dataset manifests: turn raw corpora into speaker-disjoint train/val/test splits.

Why this file exists
--------------------
Almost every student wake-word repo reports ~99% accuracy on a *random* split.
That number is meaningless: the same speaker appears in train and test, so the
model memorises the voice, not the word.  Every split produced here is
**speaker-disjoint** and the evaluation in ``ml/tools/evaluate.py`` refuses to
report accuracy on a leaking manifest.

Corpora
-------
hi_bixby (curated, the positive class)
    ``positive/positive_NNN.wav``  - 3 speakers, 400 clips each, half clean and
    half recorded over real background noise.  The author ranges are documented
    in the upstream README and encoded in :data:`HI_BIXBY_POSITIVE_AUTHORS`.
    ``negative/negative_NNN.wav``  - two very different kinds of negative:
      * 1-600   "fuzzy words" (``Hi Vicky``, ``Hi Dixie``, ...) - 10 phrases per
        author block that are phonetically confusable with the keyword.  These
        are the *hard negatives* and they are the reason this corpus is worth
        using: a model that survives them will not fire on the TV.
      * 601-800 real urban background recordings (UrbanSound8K).  Half are used
        as an augmentation noise bank, half are held out for evaluation.

gsc (Google Speech Commands v0.02, the negative class)
    ~2000 clips per word across ~2000 speakers; filenames are
    ``<speaker>_nohash_<n>.wav`` so speaker-disjoint splits are trivial.
    35 words / 41,727 clips are cached locally.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.common.audio import load_audio, speaker_id_from_gsc  # noqa: E402
from ml.data.fetch_datasets import DEFAULT_DATA_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus layout constants (from the upstream README - do not guess these)
# ---------------------------------------------------------------------------
#: (inclusive start, inclusive end, author) for ``positive/positive_NNN.wav``
HI_BIXBY_POSITIVE_AUTHORS = [
    (1, 200, "hb_a1"),
    (201, 400, "hb_a2"),
    (401, 600, "hb_a3"),
    (601, 800, "hb_a2"),
    (801, 1000, "hb_a1"),
    (1001, 1200, "hb_a3"),
]
#: Speakers whose keywords are used for TRAINING.  ``hb_a3`` is held out.
HI_BIXBY_TRAIN_AUTHORS = ("hb_a1", "hb_a2")
HI_BIXBY_TEST_AUTHOR = "hb_a3"

#: "Fuzzy word" blocks, one speaker per block.
HI_BIXBY_FUZZY_AUTHORS = [(1, 200, "hb_a1"), (201, 400, "hb_a2"), (401, 600, "hb_a3")]
#: Real background recordings: 601-700 feed the augmentation noise bank,
#: 701-800 are held out so that evaluation uses noise the model never heard.
HI_BIXBY_NOISE_BANK_RANGE = (601, 700)
HI_BIXBY_NOISE_TEST_RANGE = (701, 800)

#: The ten confusable phrases, for reporting per-phrase false-activation rates.
FUZZY_PHRASES = [
    "Hi Fix Me", "Hi Bixy", "Hi Vicky", "Hi Ixy", "Hi Pixie",
    "Hi Ricky", "Hi Dixie", "Hi Trixy", "Hi Nixie", "Hi Lexy",
]

#: GSC words deliberately included because the Speech Commands paper itself
#: treats them as the "unknown/filler" words.
GSC_FILLER_WORDS = ("marvin", "sheila")


@dataclass
class ClipRecord:
    path: str
    label: int  # 1 = keyword, 0 = not keyword
    speaker: str
    source: str  # "hi_bixby" | "gsc" | "hi_bixby_noise"
    kind: str  # "keyword" | "fuzzy" | "background" | "gsc_word"
    detail: str  # fuzzy phrase or GSC word
    split: str  # "train" | "val" | "test" | "noise_bank"
    duration_s: float


def _clip_index(path: Path) -> int:
    """``positive_042.wav`` -> ``42``."""
    return int(path.stem.split("_")[-1])


def _author_for(index: int, ranges) -> str:
    for start, end, author in ranges:
        if start <= index <= end:
            return author
    return "unknown"


def _fuzzy_phrase(index_within_block: int) -> str:
    """Map a 1..200 index inside a fuzzy block to its phrase."""
    return FUZZY_PHRASES[(index_within_block - 1) // 20]


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------
def build_hi_bixby_manifest(root: Path, val_fraction: float = 0.15, seed: int = 1234) -> list[ClipRecord]:
    """Positives + fuzzy hard negatives + the two background-noise banks."""
    rng = random.Random(seed)
    records: list[ClipRecord] = []

    # -- positives --------------------------------------------------------
    for wav in sorted((root / "positive").glob("*.wav")):
        idx = _clip_index(wav)
        author = _author_for(idx, HI_BIXBY_POSITIVE_AUTHORS)
        if author in HI_BIXBY_TRAIN_AUTHORS:
            # Carve the validation set out of the *training speakers* so that
            # the test speaker is never used for model selection either.
            split = "val" if rng.random() < val_fraction else "train"
        else:
            split = "test"
        noisy = idx >= 601
        records.append(
            ClipRecord(
                path=str(wav.relative_to(root.parent)),
                label=1,
                speaker=author,
                source="hi_bixby",
                kind="keyword",
                detail="hi bixby" + (" (noisy)" if noisy else " (clean)"),
                split=split,
                duration_s=0.0,
            )
        )

    # -- fuzzy hard negatives --------------------------------------------
    for wav in sorted((root / "negative").glob("*.wav")):
        idx = _clip_index(wav)
        if HI_BIXBY_NOISE_BANK_RANGE[0] <= idx <= HI_BIXBY_NOISE_TEST_RANGE[1]:
            # background recordings, handled separately below
            split = (
                "noise_bank"
                if idx <= HI_BIXBY_NOISE_BANK_RANGE[1]
                else "noise_test"
            )
            records.append(
                ClipRecord(
                    path=str(wav.relative_to(root.parent)),
                    label=0,
                    speaker="urbansound8k",
                    source="hi_bixby_noise",
                    kind="background",
                    detail="urban background",
                    split=split,
                    duration_s=0.0,
                )
            )
            continue
        author = _author_for(idx, HI_BIXBY_FUZZY_AUTHORS)
        if author in HI_BIXBY_TRAIN_AUTHORS:
            split = "val" if rng.random() < val_fraction else "train"
        else:
            split = "test"
        block_index = ((idx - 1) % 200) + 1
        records.append(
            ClipRecord(
                path=str(wav.relative_to(root.parent)),
                label=0,
                speaker=author,
                source="hi_bixby",
                kind="fuzzy",
                detail=_fuzzy_phrase(block_index),
                split=split,
                duration_s=0.0,
            )
        )

    return records


def build_gsc_manifest(
    root: Path,
    words: list[str],
    max_train: int = 12_000,
    max_val: int = 1_200,
    max_test: int = 1_500,
    seed: int = 7,
) -> list[ClipRecord]:
    """Speaker-disjoint GSC negatives.

    Speakers are assigned to train/val/test first, then clips are drawn from
    those speakers.  Assigning clips first and speakers later would leak.
    """
    rng = random.Random(seed)

    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for word in words:
        wdir = root / word
        if not wdir.is_dir():
            continue
        for wav in sorted(wdir.glob("*.wav")):
            by_speaker[speaker_id_from_gsc(wav.name)].append(wav)

    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    n = len(speakers)
    n_val = max(1, int(n * 0.10))
    n_test = max(1, int(n * 0.12))
    assign = {}
    for i, spk in enumerate(speakers):
        assign[spk] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    caps = {"train": max_train, "val": max_val, "test": max_test}
    counts = Counter()
    records: list[ClipRecord] = []
    # Interleave words round-robin so no split is dominated by a single word.
    per_split: dict[str, dict[str, list[Path]]] = {s: defaultdict(list) for s in caps}
    for spk, paths in by_speaker.items():
        split = assign[spk]
        for p in paths:
            per_split[split][p.parent.name].append(p)

    for split, cap in caps.items():
        words_in_split = sorted(per_split[split])
        if not words_in_split:
            continue
        cursor = {w: 0 for w in words_in_split}
        while counts[split] < cap:
            progressed = False
            for word in words_in_split:
                if counts[split] >= cap:
                    break
                lst = per_split[split][word]
                if cursor[word] < len(lst):
                    p = lst[cursor[word]]
                    cursor[word] += 1
                    records.append(
                        ClipRecord(
                            path=str(p.relative_to(root.parent)),
                            label=0,
                            speaker=speaker_id_from_gsc(p.name),
                            source="gsc",
                            kind="gsc_word",
                            detail=word,
                            split=split,
                            duration_s=p.stat().st_size / 32000.0,  # approx 16k mono 16-bit
                        )
                    )
                    counts[split] += 1
                    progressed = True
            if not progressed:
                break

    return records


# ---------------------------------------------------------------------------
# GSC-as-keyword mode (proves the pipeline retargets to ANY custom keyword)
# ---------------------------------------------------------------------------
def build_gsc_keyword_manifest(
    root: Path,
    keyword: str,
    other_words: list[str],
    seed: int = 11,
) -> list[ClipRecord]:
    """Treat an arbitrary GSC word as the custom keyword.

    SIH26172 requires the model to work for *a given custom keyword* - the
    keyword is assigned on the day, not chosen by us.  This mode trains and
    evaluates the identical pipeline on a different keyword so the claim
    "we can retarget on demand" is demonstrated, not asserted.
    """
    records = build_gsc_manifest(root, other_words, max_train=10_000, max_val=1_000, max_test=1_200, seed=seed)
    for rec in records:
        rec.kind = "gsc_other"
    kw_dir = root / keyword
    if not kw_dir.is_dir():
        raise FileNotFoundError(f"keyword '{keyword}' not found under {root}")

    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for wav in sorted(kw_dir.glob("*.wav")):
        # GSC has ~4 clips per speaker per word: keep the whole speaker together.
        by_speaker[speaker_id_from_gsc(wav.name)].append(wav)

    speakers = sorted(by_speaker)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * 0.10))
    n_test = max(1, int(len(speakers) * 0.12))
    for i, spk in enumerate(speakers):
        split = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
        for wav in by_speaker[spk]:
            records.append(
                ClipRecord(
                    path=str(wav.relative_to(root.parent)),
                    label=1,
                    speaker=spk,
                    source="gsc",
                    kind="keyword",
                    detail=keyword,
                    split=split,
                    duration_s=1.0,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def fill_durations(records: list[ClipRecord], data_dir: Path, limit: int | None = None) -> None:
    """Measure real durations (needed for the streaming simulator and for the
    "clip fully inside the window" logic)."""
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        if rec.duration_s > 0.0:
            continue
        p = data_dir / rec.path
        try:
            from ml.common.audio import wav_duration

            rec.duration_s = wav_duration(p)
        except Exception:
            rec.duration_s = 1.0


def summarize(records: list[ClipRecord]) -> dict:
    out: dict = {"total": len(records), "splits": {}, "speaker_disjoint": {}}
    for split in ("train", "val", "test"):
        sel = [r for r in records if r.split == split]
        key = Counter((r.kind) for r in sel)
        out["splits"][split] = {"clips": len(sel), "kinds": dict(key)}
    # Verify no speaker leaks across splits.
    spk = defaultdict(set)
    for r in records:
        if r.split in ("train", "val", "test"):
            spk[r.source].add((r.speaker, r.split))
    leaks = []
    for src, pairs in spk.items():
        by_split: dict[str, set[str]] = defaultdict(set)
        for speaker, split in pairs:
            by_split[split].add(speaker)
        for a in ("train", "val"):
            overlap = by_split[a] & by_split["test"]
            if overlap:
                leaks.append({"source": src, "train_or_val": a, "speakers": sorted(overlap)[:5]})
    out["speaker_disjoint"] = {"ok": not leaks, "leaks": leaks}
    return out


def portable_path(path: Path) -> str:
    """Render ``path`` for a committed manifest.

    The manifest is checked into Git, so an absolute path from the machine that
    built it (``/home/someone/.cache/...``) is both noise for a reviewer and wrong
    for anyone else.  The field is provenance only - nothing reads it back - so it
    is written relative to the home directory when it lives there.
    """
    try:
        rel = path.resolve().relative_to(Path.home())
    except (ValueError, RuntimeError):
        return str(path)
    return f"~/{rel}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out", type=Path, default=Path("ml/data/manifests"))
    ap.add_argument("--keyword", default="hi_bixby", help="hi_bixby | <gsc word>")
    ap.add_argument("--max-train-neg", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-duration-probe", action="store_true")
    args = ap.parse_args(argv)

    data = args.data_dir / "raw"
    hb_root = data / "hi_bixby"
    gsc_root = data / "gsc"
    if not hb_root.is_dir():
        print(f"ERROR: {hb_root} missing - run: python -m ml.data.fetch_datasets", file=sys.stderr)
        return 2

    gsc_words = sorted(p.name for p in gsc_root.iterdir() if p.is_dir()) if gsc_root.is_dir() else []
    if not gsc_words:
        print("WARNING: no GSC negatives cached; the model will overfit the 800 fuzzy negatives", file=sys.stderr)

    if args.keyword == "hi_bixby":
        records = build_hi_bixby_manifest(hb_root, seed=args.seed)
        records += build_gsc_manifest(gsc_root, gsc_words, max_train=args.max_train_neg, seed=args.seed)
        manifest_name = "hi_bixby.json"
    else:
        if args.keyword not in gsc_words:
            print(f"ERROR: keyword '{args.keyword}' is not cached in {gsc_root}", file=sys.stderr)
            return 2
        others = [w for w in gsc_words if w != args.keyword]
        records = build_gsc_keyword_manifest(gsc_root, args.keyword, others, seed=args.seed)
        records += build_hi_bixby_manifest(hb_root, seed=args.seed)
        manifest_name = f"gsc_{args.keyword}.json"

    if not args.no_duration_probe:
        fill_durations(records, data)

    summary = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / manifest_name
    payload = {
        "keyword": args.keyword,
        "seed": args.seed,
        "data_dir": portable_path(args.data_dir / "raw"),
        "summary": summary,
        "clips": [asdict(r) for r in records],
    }
    out_path.write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path} ({len(records)} clips)")
    if not summary["speaker_disjoint"]["ok"]:
        print("ERROR: speaker leakage detected between train/val and test!", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
