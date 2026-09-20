"""Telemetry storage (SQLite).

The dashboard has to show *something real*: wake events with their on-device
score, the µ-law uplink size, the ASR decode time and the transcript.  All of it
lands here, one row per event, and the aggregate queries below are what the
metrics panel renders.

SQLite is deliberate: zero setup, single file, and ``make clean-data`` removes it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS wake_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT    NOT NULL,
    ts            REAL    NOT NULL,
    keyword       TEXT,
    score         REAL,
    threshold     REAL,
    latency_ms    REAL,
    policy        TEXT,
    firmware      TEXT,
    client        TEXT,
    meta          TEXT
);
CREATE INDEX IF NOT EXISTS idx_wake_ts ON wake_events(ts DESC);

CREATE TABLE IF NOT EXISTS asr_requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id     TEXT,
    ts            REAL   NOT NULL,
    codec         TEXT,
    bytes_in      INTEGER,
    audio_seconds REAL,
    decode_ms     REAL,
    text          TEXT,
    backend       TEXT,
    rtf           REAL
);
CREATE INDEX IF NOT EXISTS idx_asr_ts ON asr_requests(ts DESC);

CREATE TABLE IF NOT EXISTS stream_sessions (
    session_id    TEXT PRIMARY KEY,
    device_id     TEXT,
    started_ts    REAL,
    ended_ts      REAL,
    frames        INTEGER DEFAULT 0,
    bytes_in      INTEGER DEFAULT 0,
    detections    INTEGER DEFAULT 0
);
"""


@dataclass
class Store:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- writes -----------------------------------------------------------
    def add_wake_event(
        self,
        device_id: str,
        keyword: str | None,
        score: float | None,
        threshold: float | None,
        latency_ms: float | None,
        policy: str | None = None,
        firmware: str | None = None,
        client: str | None = None,
        meta: dict | None = None,
        ts: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO wake_events (device_id, ts, keyword, score, threshold, latency_ms, policy, firmware, client, meta)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                device_id,
                ts if ts is not None else time.time(),
                keyword,
                score,
                threshold,
                latency_ms,
                policy,
                firmware,
                client,
                json.dumps(meta or {}),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_asr_request(
        self,
        device_id: str | None,
        codec: str,
        bytes_in: int,
        audio_seconds: float,
        decode_ms: float,
        text: str,
        backend: str,
        rtf: float,
        ts: float | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO asr_requests (device_id, ts, codec, bytes_in, audio_seconds, decode_ms, text, backend, rtf)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                device_id,
                ts if ts is not None else time.time(),
                codec,
                bytes_in,
                audio_seconds,
                decode_ms,
                text,
                backend,
                rtf,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_session(self, session_id: str, device_id: str | None, frames: int, bytes_in: int, detections: int) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT INTO stream_sessions (session_id, device_id, started_ts, ended_ts, frames, bytes_in, detections)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(session_id) DO UPDATE SET"
            "   ended_ts=excluded.ended_ts,"
            "   frames=stream_sessions.frames + excluded.frames,"
            "   bytes_in=stream_sessions.bytes_in + excluded.bytes_in,"
            "   detections=stream_sessions.detections + excluded.detections",
            (session_id, device_id, now, now, frames, bytes_in, detections),
        )
        self._conn.commit()

    # -- reads ------------------------------------------------------------
    def recent_wake_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM wake_events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d.get("meta") or "{}")
            except Exception:
                d["meta"] = {}
            out.append(d)
        return out

    def recent_asr_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM asr_requests ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def metrics(self) -> dict[str, Any]:
        """Aggregates for the dashboard: counts, bandwidth, latency, RTF."""
        wake = self._conn.execute(
            "SELECT COUNT(*) n, AVG(score) avg_score, AVG(latency_ms) avg_latency,"
            " MAX(latency_ms) max_latency FROM wake_events"
        ).fetchone()
        asr = self._conn.execute(
            "SELECT COUNT(*) n, SUM(bytes_in) bytes, SUM(audio_seconds) audio_s,"
            " AVG(decode_ms) avg_decode_ms, AVG(rtf) avg_rtf FROM asr_requests"
        ).fetchone()
        sessions = self._conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(frames),0) frames, COALESCE(SUM(bytes_in),0) bytes,"
            " COALESCE(SUM(detections),0) detections FROM stream_sessions"
        ).fetchone()
        devices = self._conn.execute("SELECT COUNT(DISTINCT device_id) n FROM wake_events").fetchone()
        latency_rows = self._conn.execute(
            "SELECT latency_ms FROM wake_events WHERE latency_ms IS NOT NULL ORDER BY latency_ms"
        ).fetchall()
        latencies = [float(r["latency_ms"]) for r in latency_rows]

        def pct(p: float) -> float | None:
            if not latencies:
                return None
            idx = min(len(latencies) - 1, max(0, int(round(p / 100.0 * (len(latencies) - 1)))))
            return latencies[idx]

        audio_s = float(asr["audio_s"] or 0.0)
        bytes_in = int(asr["bytes"] or 0)
        pcm16_bytes = audio_s * 16000 * 2  # what 16 kHz linear PCM would have cost
        return {
            "wake_events": int(wake["n"] or 0),
            "devices": int(devices["n"] or 0),
            "avg_wake_score": float(wake["avg_score"]) if wake["avg_score"] is not None else None,
            "latency_ms": {
                "avg": float(wake["avg_latency"]) if wake["avg_latency"] is not None else None,
                "p50": pct(50),
                "p90": pct(90),
                "max": float(wake["max_latency"]) if wake["max_latency"] is not None else None,
                "samples": len(latencies),
            },
            "asr": {
                "requests": int(asr["n"] or 0),
                "audio_seconds": audio_s,
                "bytes_in": bytes_in,
                "avg_decode_ms": float(asr["avg_decode_ms"]) if asr["avg_decode_ms"] is not None else None,
                "avg_rtf": float(asr["avg_rtf"]) if asr["avg_rtf"] is not None else None,
            },
            "uplink_savings": {
                "codec": "g711_mulaw_8khz",
                "bitrate_kbps": 64.0,
                "pcm16_bitrate_kbps": 256.0,
                "bytes_sent": bytes_in,
                "bytes_if_pcm16": int(pcm16_bytes),
                "bytes_saved": max(0, int(pcm16_bytes) - bytes_in),
                "reduction_ratio": (1.0 - bytes_in / pcm16_bytes) if pcm16_bytes > 0 else None,
            },
            "streams": {
                "sessions": int(sessions["n"] or 0),
                "frames": int(sessions["frames"] or 0),
                "bytes_in": int(sessions["bytes"] or 0),
                "detections": int(sessions["detections"] or 0),
            },
        }

    def clear(self, tables: Iterable[str] = ("wake_events", "asr_requests", "stream_sessions")) -> None:
        for t in tables:
            self._conn.execute(f"DELETE FROM {t}")  # noqa: S608 - fixed identifiers
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
