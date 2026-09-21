#!/usr/bin/env python3
"""Fetch the public speech corpora used to train the KWS model.

Raw audio is downloaded **outside** the repository (default
``~/.cache/kws-datasets``) so that clones stay small and no audio ever lands in
Git history.  Override with ``KWS_DATA_DIR=/path/to/scratch``.

Sources
-------
* ``hi_bixby``  - 1200 real recordings of the wake phrase *"hi bixby"* from
  3 speakers (600 clean + 600 with background noise) **plus 800 "fuzzy"
  phonetically-similar hard negatives**.  This is the positive corpus.
  License: see the upstream repo (research use).
* ``gsc``       - Google Speech Commands v0.02 (CC-BY 4.0).  Used as a broad
  negative pool: ~2000 recordings per word, tagged with a *speaker hash*,
  which lets us build speaker-disjoint train/test splits.

Both are fetched with tools that are always available in CI/sandbox images
(``git`` and ``gh``) because the raw.githubusercontent CDN is blocked in many
restricted networks while ``github.com`` / the GitHub API are not.

Usage
-----
    python -m ml.data.fetch_datasets --corpora hi_bixby gsc --words yes no up ...
    python -m ml.data.fetch_datasets --list
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("KWS_DATA_DIR", Path.home() / ".cache" / "kws-datasets"))

HI_BIXBY_REPO = "https://github.com/Nitesh-04/hi-bixby-wakeword-dataset.git"
GSC_REPO = "synesthesiam/google-speech-commands"  # compressed mirror of GSC v0.02

#: Default negative word pool.  Chosen for phonetic coverage (fricatives,
#: plosives, nasals, vowels) and for including the wake-word-like words
#: "marvin" / "sheila" that the Speech Commands paper itself uses.
DEFAULT_WORDS = [
    "yes", "no", "up", "down", "go", "stop", "left", "right",
    "marvin", "sheila", "happy", "house", "tree", "bird", "cat", "dog",
    "zero", "three", "seven", "wow",
]


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# hi_bixby
# ---------------------------------------------------------------------------
def fetch_hi_bixby(data_dir: Path, force: bool = False) -> Path:
    """Sparse-clone only the ``positive/`` and ``negative/`` folders."""
    dest = data_dir / "raw" / "hi_bixby"
    positive = dest / "positive"
    negative = dest / "negative"
    if positive.exists() and negative.exists() and any(positive.glob("*.wav")) and not force:
        log(f"hi_bixby already present ({len(list(positive.glob('*.wav')))} pos, "
            f"{len(list(negative.glob('*.wav')))} neg)")
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"cloning hi_bixby -> {dest}")
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout", HI_BIXBY_REPO, str(dest)])
    run(["git", "sparse-checkout", "init", "--cone"], cwd=dest)
    run(["git", "sparse-checkout", "set", "positive", "negative"], cwd=dest)
    run(["git", "checkout"], cwd=dest)
    # The clone metadata is ~300 MB of objects we no longer need.
    shutil.rmtree(dest / ".git", ignore_errors=True)
    n_pos = len(list(positive.glob("*.wav")))
    n_neg = len(list(negative.glob("*.wav")))
    log(f"hi_bixby ready: {n_pos} positives, {n_neg} negatives")
    if n_pos == 0:
        raise RuntimeError("hi_bixby download produced no positives")
    return dest


# ---------------------------------------------------------------------------
# Google Speech Commands (compressed mirror, one tarball per word)
# ---------------------------------------------------------------------------
def _blobless_sparse_clone(repo: str, paths: list[str], work: Path) -> Path:
    """Clone only ``paths`` out of a GitHub repo (works where raw CDNs are blocked)."""
    shutil.rmtree(work, ignore_errors=True)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         f"https://github.com/{repo}.git", str(work)])
    run(["git", "sparse-checkout", "set", "--no-cone", *paths], cwd=work)
    run(["git", "checkout"], cwd=work)
    return work


def fetch_gsc(data_dir: Path, words: list[str], force: bool = False) -> dict[str, int]:
    """Download the requested GSC words (one bulk clone), extract them, return counts."""
    root = data_dir / "raw" / "gsc"
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    todo = []
    for word in words:
        out_dir = root / word
        if out_dir.exists() and any(out_dir.glob("*.wav")) and not force:
            counts[word] = len(list(out_dir.glob("*.wav")))
            log(f"gsc/{word} already present ({counts[word]} clips)")
        else:
            todo.append(word)

    if todo:
        work = data_dir / "raw" / ".gsc_clone"
        log(f"cloning {len(todo)} word archives from {GSC_REPO} (blobless sparse clone)")
        try:
            _blobless_sparse_clone(GSC_REPO, [f"{w}.tar.gz" for w in todo], work)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network dependent
            log(f"git clone failed ({exc.stderr.strip()[:200]}); falling back to gh api")
            work.mkdir(parents=True, exist_ok=True)
            for word in todo:
                subprocess.run(
                    ["gh", "api", f"repos/{GSC_REPO}/contents/{word}.tar.gz",
                     "-H", "Accept: application/vnd.github.raw"],
                    stdout=open(work / f"{word}.tar.gz", "wb"), check=True,
                )

        for word in todo:
            src = work / f"{word}.tar.gz"
            if not src.exists():
                log(f"WARNING: {word}.tar.gz missing after fetch")
                continue
            kept = root / f"{word}.tar.gz"
            if not kept.exists():
                shutil.copyfile(src, kept)  # keep the archive so re-runs are cheap
            out_dir = root / word
            out_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(src, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(".wav")]
                for m in members:
                    m.name = Path(m.name).name  # tars are flat; normalise anyway
                tf.extractall(out_dir, members=members)
            counts[word] = len(list(out_dir.glob("*.wav")))
            log(f"gsc/{word}: extracted {counts[word]} clips")
        shutil.rmtree(work, ignore_errors=True)

    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def available_words(data_dir: Path) -> list[str]:
    return sorted(p.stem for p in (data_dir / "raw" / "gsc").glob("*.tar.gz"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=["hi_bixby", "gsc"], choices=["hi_bixby", "gsc"])
    ap.add_argument("--words", nargs="+", default=DEFAULT_WORDS, help="GSC words to fetch as negatives")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--list", action="store_true", help="list what is already cached and exit")
    args = ap.parse_args(argv)

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        print(f"data dir: {data_dir}")
        for sub in ("raw/hi_bixby/positive", "raw/hi_bixby/negative"):
            p = data_dir / sub
            print(f"  {sub:32s} {len(list(p.glob('*.wav'))) if p.exists() else 0} clips")
        gsc_root = data_dir / "raw" / "gsc"
        words = sorted(p.name for p in gsc_root.glob("*") if p.is_dir())
        print(f"  raw/gsc/{'':24s} {len(words)} words: {' '.join(words)}")
        return 0

    t0 = time.time()
    if "hi_bixby" in args.corpora:
        fetch_hi_bixby(data_dir, force=args.force)
    if "gsc" in args.corpora:
        try:
            counts = fetch_gsc(data_dir, list(args.words), force=args.force)
            log(f"gsc negatives available: {sum(counts.values())} clips across {len(counts)} words")
        except Exception as exc:  # a partial negative pool still trains a model
            log(f"WARNING: gsc fetch incomplete: {exc}")

    log(f"done in {time.time() - t0:.1f}s -> {data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
