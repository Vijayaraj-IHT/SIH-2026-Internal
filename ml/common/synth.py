"""Continuous-stream synthesis: real clips, real noise, known ground truth.

Used by
* ``ml/tools/streaming_eval.py`` - to benchmark latency and false activations, and
* ``server/demo.py``              - to drive the dashboard's end-to-end demo.

Keeping one implementation means the latency the dashboard animates is produced
by exactly the same stream construction the benchmark measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.common.audio import SAMPLE_RATE, float_from_pcm16

# ---------------------------------------------------------------------------
# Clip containers
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    """One corpus utterance with its annotated speech boundaries."""

    audio: np.ndarray  # float32, 16 kHz mono
    onset: int  # first sample of speech
    offset: int  # one past the last sample of the keyword
    kind: str  # keyword | fuzzy | gsc_word | background
    detail: str
    speaker: str

    @property
    def speech_samples(self) -> int:
        return max(1, self.offset - self.onset)

    @property
    def speech_ms(self) -> float:
        return self.speech_samples / SAMPLE_RATE * 1000.0


@dataclass
class Event:
    """A keyword occurrence placed into a stream."""

    t_end_s: float  # stream time at which the keyword ends (ground truth)
    clip: Clip
    snr_db: float | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_clips(cache_dir: Path, split: str, kind: str | None = None, limit: int | None = None) -> list[Clip]:
    """Read clips out of the TFRecord cache (train/val/test/noise_* splits)."""
    import tensorflow as tf

    from ml.data.build_cache import FEATURE_SPEC

    clips: list[Clip] = []
    for path in sorted(Path(cache_dir).glob(f"{split}-*.tfrec")):
        for raw in tf.data.TFRecordDataset([str(path)]):
            feat = tf.io.parse_single_example(raw, FEATURE_SPEC)
            k = feat["kind"].numpy().decode()
            if kind is not None and k != kind:
                continue
            pcm = tf.io.decode_raw(feat["audio"], tf.int16).numpy()
            clips.append(
                Clip(
                    audio=float_from_pcm16(pcm),
                    onset=int(feat["onset"].numpy()),
                    offset=int(feat["offset"].numpy()),
                    kind=k,
                    detail=feat["detail"].numpy().decode(),
                    speaker=feat["speaker"].numpy().decode(),
                )
            )
            if limit is not None and len(clips) >= limit:
                return clips
    return clips


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------
def _power(x: np.ndarray) -> float:
    return float(np.mean(np.square(x)) + 1e-12)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` at the requested SNR (speech power / noise power)."""
    noise = np.resize(noise, speech.size) if noise.size < speech.size else noise[: speech.size]
    gain = np.sqrt(_power(speech) / (_power(noise) * (10.0 ** (snr_db / 10.0))))
    return speech + noise * gain


def tile_background(backgrounds: list[Clip], total: int, rng: np.random.Generator, level: float = 0.15) -> np.ndarray:
    """A continuous, never-silent background bed."""
    if not backgrounds or total <= 0:
        return np.zeros(total, dtype=np.float32)
    chunk = int(rng.uniform(2.0, 4.0) * SAMPLE_RATE)
    tiles = []
    n_tiles = total // max(chunk, 1) + 1
    for _ in range(n_tiles):
        b = backgrounds[int(rng.integers(0, len(backgrounds)))].audio
        b = np.resize(b, chunk) if b.size < chunk else b[:chunk]
        tiles.append(b)
    return (np.concatenate(tiles)[:total] * level).astype(np.float32)


# ---------------------------------------------------------------------------
# Stream assembly
# ---------------------------------------------------------------------------
def build_stream(
    keywords: list[Clip],
    negatives: list[Clip],
    backgrounds: list[Clip],
    rng: np.random.Generator,
    duration_s: float,
    keyword_probability: float = 0.45,
    keyword_spacing_s: tuple[float, float] = (4.0, 9.0),
    negative_spacing_s: tuple[float, float] = (1.5, 4.0),
    snr_choices: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0),
    include_keywords: bool = True,
    include_negatives: bool = True,
    add_background_bed: bool = True,
) -> tuple[np.ndarray, list[Event]]:
    """Synthesise a continuous audio stream plus the ground truth it contains.

    Keyword events are placed at known positions, mixed with real background
    recordings at a randomly chosen SNR from ``snr_choices`` (including 0 dB, so
    the benchmark includes genuinely hard conditions), and separated by realistic
    gaps.  The returned ``Event.t_end_s`` is the ground truth used for latency.
    """
    total = int(duration_s * SAMPLE_RATE)
    stream = np.zeros(total, dtype=np.float32)
    events: list[Event] = []

    if add_background_bed:
        stream += tile_background(backgrounds, total, rng)

    def add_clip(clip: Clip, at: int, snr_db: float | None) -> None:
        segment = clip.audio
        if snr_db is not None and backgrounds:
            bed = backgrounds[int(rng.integers(0, len(backgrounds)))].audio
            segment = mix_at_snr(segment, bed, snr_db)
        end = min(total, at + segment.size)
        if end > at:
            stream[at:end] += segment[: end - at]

    cursor = int(rng.uniform(0.5, 2.0) * SAMPLE_RATE)
    while cursor < total - SAMPLE_RATE:
        use_keyword = include_keywords and keywords and rng.uniform() < keyword_probability
        if use_keyword:
            clip = keywords[int(rng.integers(0, len(keywords)))]
            snr = float(snr_choices[int(rng.integers(0, len(snr_choices)))])
            add_clip(clip, cursor, snr)
            if clip.offset < clip.audio.size:
                events.append(Event(t_end_s=(cursor + clip.offset) / SAMPLE_RATE, clip=clip, snr_db=snr))
            cursor += int(rng.uniform(*keyword_spacing_s) * SAMPLE_RATE)
        elif include_negatives and negatives:
            clip = negatives[int(rng.integers(0, len(negatives)))]
            snr = float(snr_choices[int(rng.integers(0, len(snr_choices)))])
            add_clip(clip, cursor, snr)
            cursor += int(rng.uniform(*negative_spacing_s) * SAMPLE_RATE)
        else:
            cursor += int(rng.uniform(1.0, 3.0) * SAMPLE_RATE)

    peak = float(np.max(np.abs(stream))) if stream.size else 1.0
    if peak > 1.0:
        stream = stream / peak
    return stream.astype(np.float32), events
