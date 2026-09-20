"""Remote ASR back-ends for the post-wake audio stream.

SIH26172 splits the work in two: the *edge* decides when to wake, the *server*
does the heavy speech recognition.  This module is the server half, behind a
small adapter interface so the demo never depends on a single vendor:

======================  ==========================================  ===========
backend                 requirements                                notes
======================  ==========================================  ===========
``openai``              ``OPENAI_API_KEY``                          needs network
``pocketsphinx``        ``pip install pocketsphinx``                fully offline
``reference``           none                                        deterministic
======================  ==========================================  ===========

``reference`` is not a pretend model: it performs a real, deterministic
utterance-level match against the keyword/command inventory using a DTW distance
on log-mel features, and reports itself as ``reference`` in every response so no
one can mistake it for a full ASR system.  It exists so the streaming path,
bandwidth accounting and latency measurements can be exercised in CI without
network access or a 40 MB model download.
"""

from __future__ import annotations

import glob
import io
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.common.audio import SAMPLE_RATE, float_from_pcm16, pcm16_from_float, resample, ulaw_decode


@dataclass
class AsrResult:
    text: str
    backend: str
    audio_seconds: float
    decode_seconds: float
    confidence: float | None = None

    @property
    def real_time_factor(self) -> float:
        """RTF = decode time / audio duration (< 1 means faster than real time)."""
        return self.decode_seconds / self.audio_seconds if self.audio_seconds > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "backend": self.backend,
            "audio_seconds": round(self.audio_seconds, 3),
            "decode_ms": round(self.decode_seconds * 1000, 1),
            "real_time_factor": round(self.real_time_factor, 3),
            "confidence": self.confidence,
        }


class AsrBackend:
    """Interface: 8 kHz or 16 kHz float32 mono audio in, :class:`AsrResult` out."""

    name = "base"
    available = False

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Offline: PocketSphinx
# ---------------------------------------------------------------------------
class PocketSphinxBackend(AsrBackend):
    """Fully offline open-source ASR - no API key, no network, works at the venue."""

    name = "pocketsphinx"

    def __init__(self) -> None:
        self._model_dir = None
        self._hmm = self._lm = self._dic = None
        try:
            from pocketsphinx import get_model_path  # noqa: F401

            self._locate_models()
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on install
            self.error = str(exc)
            self.available = False

    def _locate_models(self) -> None:
        """Find an acoustic model definition, tolerating the wheel's nested layout."""
        from pocketsphinx import get_model_path

        root = str(get_model_path())
        mdefs = glob.glob(os.path.join(root, "**", "mdef"), recursive=True)
        if not mdefs:
            raise FileNotFoundError(f"no acoustic model (mdef) under {root}")
        self._hmm = os.path.dirname(mdefs[0])
        lms = [p for p in glob.glob(os.path.join(root, "**", "*.lm.bin"), recursive=True) if "phone" not in p]
        if not lms:
            raise FileNotFoundError(f"no language model under {root}")
        self._lm = lms[0]
        dicts = glob.glob(os.path.join(root, "**", "cmudict*.dict"), recursive=True)
        if not dicts:
            raise FileNotFoundError(f"no dictionary under {root}")
        self._dic = dicts[0]
        self._model_dir = root

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:
        from pocketsphinx import AudioFile

        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)
        # PocketSphinx reads 16 kHz mono 16-bit PCM WAV from disk.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16_from_float(audio).tobytes())
        t0 = time.perf_counter()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fh:
            fh.write(buf.getvalue())
            fh.flush()
            decoder = AudioFile(fh.name, hmm=self._hmm, lm=self._lm, dic=self._dic)
            parts = []
            for seg in decoder:
                hyp = seg.hypothesis()
                text = getattr(hyp, "hypstr", None)
                if text is None:
                    text = str(hyp)
                if text:
                    parts.append(text.strip())
        decode_s = time.perf_counter() - t0
        return AsrResult(
            text=" ".join(parts).strip(),
            backend=self.name,
            audio_seconds=audio.size / SAMPLE_RATE,
            decode_seconds=decode_s,
        )


# ---------------------------------------------------------------------------
# Cloud: OpenAI-compatible transcription (optional)
# ---------------------------------------------------------------------------
class OpenAIBackend(AsrBackend):
    """Optional cloud back-end; selected automatically when a key is present."""

    name = "openai"

    def __init__(self, model: str = "whisper-1") -> None:
        self.model = model
        self._client = None
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            self.available = False
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=key)
            self.available = True
        except Exception:
            self.available = False

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:
        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16_from_float(audio).tobytes())
        buf.seek(0)
        buf.name = "wake.wav"  # the SDK infers the format from the name
        t0 = time.perf_counter()
        resp = self._client.audio.transcriptions.create(model=self.model, file=buf)
        return AsrResult(
            text=(getattr(resp, "text", "") or "").strip(),
            backend=self.name,
            audio_seconds=audio.size / SAMPLE_RATE,
            decode_seconds=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Reference: deterministic offline matcher (no dependencies)
# ---------------------------------------------------------------------------
class ReferenceBackend(AsrBackend):
    """Deterministic template matcher used when no real ASR is installed.

    DTW over log-mel features against a small inventory of the commands the
    keyword corpus and Speech Commands cover.  It is reported as
    ``reference`` everywhere, so the transcript is never confused with a real
    recogniser's output - but the *pipeline* (mu-law in, upsample, decode,
    timing, transcription) is exercised for real.
    """

    name = "reference"
    available = True

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        from ml.common.audio import load_audio  # noqa: F401  (kept for symmetry)

        self.vocabulary = vocabulary or [
            "hi bixby",
            "yes",
            "no",
            "stop",
            "go",
            "left",
            "right",
            "up",
            "down",
            "seven",
            "zero",
            "happy",
            "house",
            "tree",
        ]
        self.templates: dict[str, np.ndarray] = {}

    # -- feature / distance ----------------------------------------------
    @staticmethod
    def _features(x: np.ndarray) -> np.ndarray:
        from ml.common.features import FrontendParams, FrontendParams as _FP, log_mel_spectrogram

        params = _FP()
        mel = log_mel_spectrogram(x, params)
        if mel.shape[0] == 0:
            return np.zeros((1, params.num_mel_bins), dtype=np.float32)
        # per-utterance normalisation makes the matcher gain-invariant
        return ((mel - mel.mean(axis=0)) / (mel.std(axis=0) + 1e-5)).astype(np.float32)

    @staticmethod
    def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
        n, m = a.shape[0], b.shape[0]
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            ai = a[i - 1]
            for j in range(1, m + 1):
                d = float(np.mean(np.abs(ai - b[j - 1])))
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        return float(cost[n, m] / max(n, m))

    def load_templates(self, cache_dir: Path | None = None) -> int:
        """Build templates from the cached corpus when it is available."""
        if cache_dir is None or not Path(cache_dir).exists():
            return 0
        try:
            from ml.common.synth import load_clips
        except Exception:
            return 0
        wanted = {v for v in self.vocabulary if " " not in v}
        for split in ("test", "val"):
            for kind in ("gsc_word", "fuzzy", "keyword"):
                try:
                    clips = load_clips(Path(cache_dir), split, kind)
                except Exception:
                    continue
                for clip in clips:
                    key = clip.detail.lower()
                    if key in wanted and key not in self.templates:
                        self.templates[key] = self._features(clip.audio[clip.onset : clip.offset])
        return len(self.templates)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> AsrResult:
        t0 = time.perf_counter()
        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)
        feats = self._features(audio)
        best, best_d = "", np.inf
        for word, tmpl in self.templates.items():
            d = self._dtw_distance(feats, tmpl)
            if d < best_d:
                best, best_d = word, d
        text = best if best and best_d < 1.5 else ""
        return AsrResult(
            text=text,
            backend=self.name,
            audio_seconds=audio.size / SAMPLE_RATE,
            decode_seconds=time.perf_counter() - t0,
            confidence=float(1.0 / (1.0 + best_d)) if best else 0.0,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BUILDERS = {
    "openai": OpenAIBackend,
    "pocketsphinx": PocketSphinxBackend,
    "reference": ReferenceBackend,
}


def build_backend(name: str = "auto", cache_dir: Path | None = None) -> AsrBackend:
    """Instantiate the best available backend.

    ``auto`` prefers the cloud back-end when a key is configured (best accuracy
    for a demo), then the offline PocketSphinx install, then the deterministic
    reference matcher.  The chosen backend is always reported in API responses
    and in ``/v1/health``.
    """
    order = [name] if name != "auto" else ["openai", "pocketsphinx", "reference"]
    for candidate in order:
        builder = _BUILDERS.get(candidate)
        if builder is None:
            continue
        try:
            backend = builder()
        except Exception:
            continue
        if getattr(backend, "available", False):
            if isinstance(backend, ReferenceBackend):
                backend.load_templates(cache_dir)
            return backend
    return ReferenceBackend()


def transcribe_ulaw(payload: bytes, backend: AsrBackend, upload_sr: int, decode_sr: int) -> AsrResult:
    """Decode the device's mu-law uplink and run ASR on it.

    The device sends 8 kHz G.711 mu-law (64 kbit/s) instead of 16 kHz linear PCM
    (256 kbit/s): a 4x reduction in the data the radio has to carry, at a speech
    quality that every telephony-trained recogniser expects.  The ASR models we
    use here want 16 kHz, so the stream is upsampled before decoding.
    """
    pcm = ulaw_decode(np.frombuffer(payload, dtype=np.uint8))
    audio = float_from_pcm16(pcm)
    if upload_sr != decode_sr:
        audio = resample(audio, upload_sr, decode_sr)
    return backend.transcribe(audio, decode_sr)
