"""End-to-end demo sessions for the dashboard.

What the demo does
------------------
It builds a continuous audio stream out of *real held-out corpus clips* mixed
with real background recordings, runs the shipped int8 model over the whole
stream, and hands the browser two things:

* ``scores`` - the per-20 ms decision curve the model actually produced, with
  the threshold applied, so the UI can animate the sliding window and show the
  instant the detector crossed it, and
* the **uplink** - the same audio as the device would send it *after* waking:
  8 kHz G.711 mu-law, 64 kbit/s, restricted to the segments that follow a
  detection.  The dashboard POSTs those bytes to ``/v1/asr/ulaw`` and displays
  the transcript, so the bandwidth saving is measured, not asserted.

Because everything is derived from a seed (plus the local clip cache), a demo is
reproducible: the same seed always yields the same stream, scores and events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.common.audio import SAMPLE_RATE, pcm16_from_float, resample, ulaw_encode
from ml.common.streaming import PolicyConfig, apply_policy
from ml.common.synth import Event, build_stream, load_clips

#: Audio sent to the ASR server after a wake event (the "listen window").
UPLINK_WINDOW_S = 5.0
#: G.711 mu-law is a telephony codec: 8 kHz, 8 bit.
UPLINK_SAMPLE_RATE = 8000


@dataclass
class DemoSession:
    """A prepared, reproducible demo stream."""

    session_id: str
    seed: int
    seconds: float
    waveform: np.ndarray  # 16 kHz float32 (what the device microphone hears)
    events: list[Event]
    scores: np.ndarray  # one per 20 ms window
    threshold: float
    policy: PolicyConfig
    keyword: str

    # -- derived -----------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return self.waveform.size / SAMPLE_RATE

    def uplink_audio(self) -> np.ndarray:
        """8 kHz float32 version of the stream (the uplink codec's input)."""
        return resample(self.waveform, SAMPLE_RATE, UPLINK_SAMPLE_RATE)

    def uplink_ulaw(self) -> bytes:
        """The whole stream as G.711 mu-law bytes (what the radio carries)."""
        return ulaw_encode(pcm16_from_float(self.uplink_audio())).tobytes()

    def uplink_slice(self, start_s: float, end_s: float) -> bytes:
        """mu-law bytes for ``[start_s, end_s)`` of the stream."""
        audio = self.uplink_audio()
        sr = UPLINK_SAMPLE_RATE
        a = max(0, int(start_s * sr))
        b = min(audio.size, int(math.ceil(end_s * sr)))
        if b <= a:
            return b""
        return ulaw_encode(pcm16_from_float(audio[a:b])).tobytes()

    def ground_truth(self) -> list[dict[str, Any]]:
        return [
            {
                "t_end_s": round(e.t_end_s, 3),
                "detail": e.clip.detail,
                "speaker": e.clip.speaker,
                "snr_db": e.snr_db,
                "speech_ms": round(e.clip.speech_ms, 1),
            }
            for e in self.events
        ]

    def to_dict(self, uplink_url: str) -> dict[str, Any]:
        durations = {
            "pcm16_16k_bits_per_second": 256_000,
            "ulaw_8k_bits_per_second": 64_000,
            "uplink_seconds": round(self.duration_s, 2),
        }
        pcm_bytes = int(self.duration_s * SAMPLE_RATE * 2)
        ulaw_bytes = int(self.duration_s * UPLINK_SAMPLE_RATE)
        return {
            "session_id": self.session_id,
            "seed": self.seed,
            "keyword": self.keyword,
            "duration_s": round(self.duration_s, 2),
            "sample_rate": SAMPLE_RATE,
            "hop_ms": 20,
            "threshold": self.threshold,
            "policy": {"confirm_windows": self.policy.confirm_windows, "refractory_ms": self.policy.refractory_ms},
            "scores": [round(float(s), 4) for s in self.scores],
            "ground_truth": self.ground_truth(),
            "uplink": {
                "url": uplink_url,
                "codec": "g711_mulaw",
                "sample_rate": UPLINK_SAMPLE_RATE,
                "window_s": UPLINK_WINDOW_S,
                "bytes_total": ulaw_bytes,
                "bytes_if_pcm16": pcm_bytes,
                "bytes_saved": max(0, pcm_bytes - ulaw_bytes),
                "reduction_ratio": round(1.0 - ulaw_bytes / pcm_bytes, 4) if pcm_bytes else None,
                **durations,
            },
        }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
class DemoBuilder:
    """Builds (and caches) demo sessions from the local clip cache."""

    def __init__(
        self,
        cache_dir: Path,
        runtime,
        max_seconds: float = 120.0,
        keyword: str = "hi bixby",
    ):
        self.cache_dir = Path(cache_dir)
        self.runtime = runtime
        self.max_seconds = max_seconds
        self.keyword = keyword
        self._clip_cache: dict[str, list] = {}
        self._sessions: dict[str, DemoSession] = {}
        self._warned: str | None = None

    # -- data -------------------------------------------------------------
    def _clips(self, split: str, kind: str | None = None, limit: int | None = None) -> list:
        key = f"{split}:{kind}:{limit}"
        if key not in self._clip_cache:
            self._clip_cache[key] = load_clips(self.cache_dir, split, kind, limit=limit)
        return self._clip_cache[key]

    @property
    def available(self) -> bool:
        """True when a real corpus cache is present (the demo needs real clips)."""
        try:
            return bool(self._clips("test", "keyword", limit=1))
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        return (
            f"no cached audio under {self.cache_dir}. Run `make data` (download + cache) "
            "to enable the stream demo; the upload/mic detector still works."
        )

    # -- session ----------------------------------------------------------
    def build(self, seconds: float = 45.0, seed: int = 2026, include_negatives: bool = True) -> DemoSession:
        session_id = f"seed{seed}-{int(seconds)}s"
        if session_id in self._sessions:
            return self._sessions[session_id]

        seconds = float(min(max(5.0, seconds), self.max_seconds))
        rng = np.random.default_rng(seed)
        keywords = self._clips("test", "keyword")
        if not keywords:
            raise RuntimeError(self.unavailable_reason())
        fuzzy = self._clips("test", "fuzzy")
        gsc = self._clips("test", "gsc_word", limit=400)
        backgrounds = self._clips("noise_test") + self._clips("noise_bank")

        waveform, events = build_stream(
            keywords=keywords,
            negatives=(fuzzy + gsc) if include_negatives else [],
            backgrounds=backgrounds,
            rng=rng,
            duration_s=seconds,
            keyword_probability=0.45,
            snr_choices=(18.0, 12.0, 8.0, 5.0),
            include_keywords=True,
            include_negatives=include_negatives,
        )

        result = self.runtime.detect(waveform)
        scores = np.asarray(result["scores"], dtype=np.float32)
        session = DemoSession(
            session_id=session_id,
            seed=seed,
            seconds=seconds,
            waveform=waveform,
            events=events,
            scores=scores,
            threshold=float(self.runtime.threshold or 0.5),
            policy=self.runtime.policy,
            keyword=self.keyword,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DemoSession | None:
        return self._sessions.get(session_id)

    def uplink_bytes(self, session_id: str, start_s: float | None = None, end_s: float | None = None) -> bytes:
        session = self.get(session_id)
        if session is None:
            return b""
        if start_s is None or end_s is None:
            return session.uplink_ulaw()
        return session.uplink_slice(start_s, end_s)
