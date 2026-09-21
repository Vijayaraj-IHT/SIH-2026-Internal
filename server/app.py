"""FastAPI application: the remote ASR server + telemetry + dashboard for SIH26172.

Endpoints
---------
``GET  /``                        dashboard (single page, no build step)
``GET  /v1/health``               liveness + which model/ASR backend is active
``GET  /v1/model``                model card (params, int8 size, MACs, metrics)
``POST /v1/wake``                 device reports a local wake-word activation
``POST /v1/asr``                  upload WAV/PCM audio, get a transcript
``POST /v1/asr/ulaw``             raw G.711 mu-law uplink (the device's real path)
``POST /v1/detect``               run the shipped int8 model on uploaded audio
``GET  /v1/demo/session``         build/replay a reproducible stream demo
``GET  /v1/demo/uplink``          binary mu-law uplink bytes for that session
``GET  /v1/telemetry``            recent wake/ASR rows
``GET  /v1/metrics``              aggregates (bandwidth saved, latency, RTF)
``POST /v1/telemetry/reset``      clear the demo database
``WS   /v1/stream``               streaming mu-law ASR with end-of-utterance VAD

The server is intentionally dependency-light: FastAPI + numpy.  The KWS model
runs on the *device*; the server only needs it to replay the demo and to serve
``/v1/detect``.
"""

from __future__ import annotations

import io
import json
import math
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.common.audio import SAMPLE_RATE, float_from_pcm16, read_wav_bytes, rms, ulaw_decode
from ml.common.streaming import PolicyConfig

from .asr import build_backend, transcribe_ulaw
from .config import REPO_ROOT, settings
from .demo import UPLINK_SAMPLE_RATE, DemoBuilder
from .runtime import KwsRuntime
from .store import Store

STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Startup: load model, ASR backend, storage
# ---------------------------------------------------------------------------
runtime = KwsRuntime(settings.run_dir, threshold_override=settings.threshold_override,
                     policy=PolicyConfig(confirm_windows=settings.confirm_windows,
                                         refractory_ms=settings.refractory_ms))
store = Store(settings.db_path)
asr_backend = build_backend(settings.asr_backend, cache_dir=settings.data_cache)
demo_builder = DemoBuilder(settings.data_cache, runtime, max_seconds=settings.demo_max_seconds,
                           keyword=(runtime.model_card.get("keyword") or "hi bixby"))

app = FastAPI(
    title="SIH26172 - Voice Activator edge server",
    description=__doc__,
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

START_TS = time.time()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class WakeEvent(BaseModel):
    device_id: str = Field(default="sim-dashboard", max_length=64)
    keyword: str | None = "hi bixby"
    score: float | None = None
    threshold: float | None = None
    latency_ms: float | None = None
    policy: str | None = None
    firmware: str | None = None
    client: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class DetectRequest(BaseModel):
    """Base64 WAV/PCM payload for /v1/detect (JSON alternative to file upload)."""

    audio_b64: str
    sample_rate: int | None = None
    confirm_windows: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_audio_bytes(data: bytes, sample_rate_hint: int | None = None) -> tuple[np.ndarray, int]:
    """Decode uploaded bytes: WAV container, raw 16-bit PCM, or mu-law."""
    if data[:4] == b"RIFF":
        return read_wav_bytes(data)
    if sample_rate_hint == 8000 or len(data) % 2 != 0:
        # odd byte count => almost certainly 8-bit mu-law
        pcm = ulaw_decode(np.frombuffer(data, dtype=np.uint8))
        return float_from_pcm16(pcm), 8000
    pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
    return float_from_pcm16(pcm), int(sample_rate_hint or settings.asr_uplink_sample_rate)


def _policy_from(confirm_windows: int | None) -> PolicyConfig:
    base = runtime.policy
    return PolicyConfig(
        threshold=base.threshold,
        confirm_windows=int(confirm_windows or base.confirm_windows),
        refractory_ms=base.refractory_ms,
    )


# ---------------------------------------------------------------------------
# Static / dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>SIH26172 server</h1><p>dashboard assets missing</p>", status_code=500)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Health / model card
# ---------------------------------------------------------------------------
@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START_TS, 1),
        "model_loaded": runtime.loaded,
        "model_error": runtime.error,
        "run_name": settings.run_name,
        "asr_backend": {"name": asr_backend.name, "available": asr_backend.available},
        "demo_available": demo_builder.available,
        "demo_unavailable_reason": None if demo_builder.available else demo_builder.unavailable_reason(),
        "uplink": {"codec": "g711_mulaw", "sample_rate": UPLINK_SAMPLE_RATE, "bitrate_kbps": 64.0},
    }


@app.get("/v1/model")
def model_card() -> dict[str, Any]:
    return runtime.info()


# ---------------------------------------------------------------------------
# Device wake notification
# ---------------------------------------------------------------------------
@app.post("/v1/wake")
def wake(event: WakeEvent, request: Request) -> dict[str, Any]:
    """Called by the edge device the moment the local detector fires.

    The response tells the device how to send the utterance (codec, sample rate,
    target window) so the contract lives on the server, not hard-coded in
    firmware that is expensive to reflash.
    """
    row_id = store.add_wake_event(
        device_id=event.device_id,
        keyword=event.keyword,
        score=event.score,
        threshold=event.threshold if event.threshold is not None else runtime.threshold,
        latency_ms=event.latency_ms,
        policy=event.policy,
        firmware=event.firmware,
        client=event.client or request.headers.get("user-agent", "unknown")[:80],
        meta=event.meta,
    )
    return {
        "event_id": row_id,
        "server_time": time.time(),
        "asr": {
            "codec": "g711_mulaw",
            "sample_rate": UPLINK_SAMPLE_RATE,
            "endpoint": "/v1/asr/ulaw",
            "max_seconds": 12,
        },
        "session_id": f"dev-{event.device_id}-{row_id}",
    }


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------
@app.post("/v1/asr")
async def asr_upload(
    file: UploadFile = File(...),
    device_id: str = Query(default="upload"),
    sample_rate: int | None = Query(default=None),
) -> dict[str, Any]:
    """Transcribe an uploaded WAV (or raw 16-bit PCM / mu-law) clip."""
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"payload too large (>{settings.max_upload_mb} MB)")
    try:
        audio, sr = _decode_audio_bytes(data, sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="empty audio")

    result = asr_backend.transcribe(audio, sr)
    store.add_asr_request(
        device_id=device_id,
        codec="wav" if data[:4] == b"RIFF" else "pcm16",
        bytes_in=len(data),
        audio_seconds=result.audio_seconds,
        decode_ms=result.decode_seconds * 1000.0,
        text=result.text,
        backend=result.backend,
        rtf=result.real_time_factor,
    )
    return {"ok": True, **result.to_dict()}


@app.post("/v1/asr/ulaw")
async def asr_ulaw(
    request: Request,
    device_id: str = Query(default="edge-device"),
    sample_rate: int = Query(default=UPLINK_SAMPLE_RATE),
) -> dict[str, Any]:
    """The device's real path: raw G.711 mu-law bytes, 8 kHz, 64 kbit/s.

    No multipart wrapping, no container overhead - the body *is* the audio.
    """
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty payload")
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="payload too large")
    result = transcribe_ulaw(payload, asr_backend, sample_rate, settings.asr_decode_sample_rate)
    store.add_asr_request(
        device_id=device_id,
        codec="g711_mulaw",
        bytes_in=len(payload),
        audio_seconds=result.audio_seconds,
        decode_ms=result.decode_seconds * 1000.0,
        text=result.text,
        backend=result.backend,
        rtf=result.real_time_factor,
    )
    pcm16_equivalent = result.audio_seconds * SAMPLE_RATE * 2
    return {
        "ok": True,
        **result.to_dict(),
        "bytes_in": len(payload),
        "bytes_if_pcm16": int(pcm16_equivalent),
        "bytes_saved": max(0, int(pcm16_equivalent) - len(payload)),
        "reduction_ratio": round(1.0 - len(payload) / pcm16_equivalent, 4) if pcm16_equivalent else None,
    }


# ---------------------------------------------------------------------------
# Local detection on uploaded audio (dashboard live demo)
# ---------------------------------------------------------------------------
@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    sample_rate: int | None = Query(default=None),
    confirm_windows: int | None = Query(default=None),
) -> dict[str, Any]:
    if not runtime.loaded:
        raise HTTPException(status_code=503, detail=f"model not loaded: {runtime.error}")
    data = await file.read()
    try:
        audio, sr = _decode_audio_bytes(data, sample_rate)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    if audio.size < 1600:
        raise HTTPException(status_code=400, detail="audio too short (need >= 0.1 s)")
    result = runtime.detect(audio, sr, policy=_policy_from(confirm_windows))
    result["level_dbfs"] = round(20.0 * math.log10(max(rms(audio), 1e-9)), 1)
    result["bytes_in"] = len(data)
    return result


@app.post("/v1/detect/json")
def detect_json(req: DetectRequest) -> dict[str, Any]:
    """Same as ``/v1/detect`` but takes base64 audio in a JSON body."""
    import base64

    if not runtime.loaded:
        raise HTTPException(status_code=503, detail=f"model not loaded: {runtime.error}")
    try:
        data = base64.b64decode(req.audio_b64, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"bad base64: {exc}") from exc
    audio, sr = _decode_audio_bytes(data, req.sample_rate)
    return runtime.detect(audio, sr, policy=_policy_from(req.confirm_windows))


# ---------------------------------------------------------------------------
# Demo stream
# ---------------------------------------------------------------------------
@app.get("/v1/demo/session")
def demo_session(seconds: float = Query(default=45.0, ge=5.0, le=120.0), seed: int = Query(default=2026)) -> dict[str, Any]:
    if not demo_builder.available:
        raise HTTPException(status_code=503, detail=demo_builder.unavailable_reason())
    if not runtime.loaded:
        raise HTTPException(status_code=503, detail=f"model not loaded: {runtime.error}")
    try:
        session = demo_builder.build(seconds=seconds, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"could not build demo: {exc}") from exc
    payload = session.to_dict(uplink_url=f"/v1/demo/uplink?session_id={session.session_id}")
    # Attach what the detector found on this stream under the deployed policy.
    from ml.common.streaming import apply_policy

    dets = apply_policy(session.scores, session.policy)
    payload["detections"] = [
        {"time_s": round(d.time_s, 3), "score": round(d.score, 4), "window_index": d.window_index} for d in dets
    ]
    return payload


@app.get("/v1/demo/uplink")
def demo_uplink(
    session_id: str = Query(...),
    start_s: float | None = Query(default=None),
    end_s: float | None = Query(default=None),
) -> Response:
    data = demo_builder.uplink_bytes(session_id, start_s, end_s)
    if not data:
        raise HTTPException(status_code=404, detail="unknown session or empty range")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Codec": "g711_mulaw",
            "X-Sample-Rate": str(UPLINK_SAMPLE_RATE),
            "X-Bytes": str(len(data)),
        },
    )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
@app.get("/v1/telemetry")
def telemetry(limit: int = Query(default=40, ge=1, le=500)) -> dict[str, Any]:
    return {
        "wake_events": store.recent_wake_events(limit),
        "asr_requests": store.recent_asr_requests(limit),
    }


@app.get("/v1/metrics")
def metrics() -> dict[str, Any]:
    return store.metrics()


@app.post("/v1/telemetry/reset")
def telemetry_reset() -> dict[str, Any]:
    store.clear()
    return {"ok": True, "cleared": True}


# ---------------------------------------------------------------------------
# WebSocket: streaming mu-law ASR
# ---------------------------------------------------------------------------
@app.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    """Bidirectional streaming ASR over G.711 mu-law.

    Protocol (JSON text control + binary audio frames):

    * client -> ``{"type":"hello","device_id":"...","sample_rate":8000}``
    * client -> binary frames of mu-law bytes (any size; 20 ms = 160 bytes typical)
    * server -> ``{"type":"ready",...}``, ``{"type":"partial","text":...}``,
      ``{"type":"final","text":...,"decode_ms":...}``, ``{"type":"stats",...}``

    End of utterance is detected with a simple energy VAD (500 ms of silence),
    which is what makes a push-to-talk-free session possible.
    """
    await ws.accept()
    device_id = "ws-client"
    sample_rate = UPLINK_SAMPLE_RATE
    buffer: list[np.ndarray] = []
    silence_ms = 0.0
    bytes_in = 0
    detections = 0
    started = time.time()
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if (text := message.get("text")) is not None:
                try:
                    ctl = json.loads(text)
                except Exception:
                    continue
                kind = ctl.get("type")
                if kind == "hello":
                    device_id = str(ctl.get("device_id", device_id))[:64]
                    sample_rate = int(ctl.get("sample_rate", sample_rate))
                    await ws.send_json(
                        {
                            "type": "ready",
                            "device_id": device_id,
                            "codec": "g711_mulaw",
                            "sample_rate": sample_rate,
                            "asr_backend": asr_backend.name,
                            "vad_silence_ms": 500,
                        }
                    )
                elif kind == "wake":
                    detections += 1
                    store.add_wake_event(
                        device_id=device_id,
                        keyword=ctl.get("keyword", "hi bixby"),
                        score=ctl.get("score"),
                        threshold=ctl.get("threshold", runtime.threshold),
                        latency_ms=ctl.get("latency_ms"),
                        policy=ctl.get("policy"),
                        meta={"via": "websocket"},
                    )
                    await ws.send_json({"type": "wake_ack", "count": detections})
                elif kind == "flush":
                    pass
                continue

            payload = message.get("bytes") or b""
            if not payload:
                continue
            bytes_in += len(payload)
            pcm = ulaw_decode(np.frombuffer(payload, dtype=np.uint8))
            audio = float_from_pcm16(pcm)
            chunk_ms = audio.size / sample_rate * 1000.0
            buffer.append(audio)

            level = rms(audio)
            if level < 0.01:  # ~-40 dBFS
                silence_ms += chunk_ms
            else:
                silence_ms = 0.0

            if silence_ms >= 500.0 and buffer:
                utterance = np.concatenate(buffer)
                buffer = []
                silence_ms = 0.0
                if utterance.size / sample_rate >= 0.2:
                    from ml.common.audio import resample as _resample

                    decoded = _resample(utterance, sample_rate, settings.asr_decode_sample_rate)
                    result = asr_backend.transcribe(decoded, settings.asr_decode_sample_rate)
                    store.add_asr_request(
                        device_id=device_id,
                        codec="g711_mulaw",
                        bytes_in=int(utterance.size),
                        audio_seconds=result.audio_seconds,
                        decode_ms=result.decode_seconds * 1000.0,
                        text=result.text,
                        backend=result.backend,
                        rtf=result.real_time_factor,
                    )
                    await ws.send_json({"type": "final", **result.to_dict()})
                    await ws.send_json(
                        {
                            "type": "stats",
                            "bytes_in": bytes_in,
                            "bytes_if_pcm16": int((time.time() - started) * 16_000 * 2),
                            "detections": detections,
                        }
                    )
            else:
                await ws.send_json({"type": "partial", "buffered_ms": round(sum(b.size for b in buffer) / sample_rate * 1000.0, 1)})
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # never let a socket error take the server down
        try:
            await ws.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        store.upsert_session(
            session_id=f"ws-{device_id}-{int(started)}",
            device_id=device_id,
            frames=detections,
            bytes_in=bytes_in,
            detections=detections,
        )


# ---------------------------------------------------------------------------
# Dev convenience
# ---------------------------------------------------------------------------
@app.get("/v1/config")
def config() -> dict[str, Any]:
    """Expose the tunables the dashboard needs to render the settings panel."""
    return {
        "run_name": settings.run_name,
        "artifacts_root": str(settings.artifacts_root),
        "cache_dir": str(settings.data_cache),
        "repo_root": str(REPO_ROOT),
        "threshold": runtime.threshold,
        "confirm_windows": runtime.policy.confirm_windows,
        "refractory_ms": runtime.policy.refractory_ms,
        "asr": {"backend": asr_backend.name, "uplink_sample_rate": UPLINK_SAMPLE_RATE},
    }


def main() -> None:  # pragma: no cover - entry point for `python -m server.app`
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
