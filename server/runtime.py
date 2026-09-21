"""Server-side KWS runtime.

The device runs the detector itself; the server still needs the same model for
three reasons:

1. the dashboard's live demo replays the exact int8 graph the firmware embeds,
2. ``/v1/detect`` lets a judge upload any audio and see the sliding-window
   decision timeline, and
3. it proves the published numbers come from the shipped artifact, not from a
   separate "demo build".

Everything is loaded from ``artifacts/<run>/`` and degrades gracefully: if the
artifacts are missing the server still starts and reports ``model_loaded: false``
with the reason, so the dashboard can tell the user to run ``make train``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.common.audio import SAMPLE_RATE, float_from_pcm16, resample, ulaw_decode
from ml.common.features import FrontendParams, QuantParams
from ml.common.streaming import (
    PolicyConfig,
    WindowScorer,
    apply_policy,
    detection_time_s,
    score_waveform_offline,
)


class KwsRuntime:
    """Loads the exported int8 model and exposes detection helpers."""

    def __init__(self, run_dir: Path, threshold_override: float | None = None, policy: PolicyConfig | None = None):
        self.run_dir = Path(run_dir)
        self.loaded = False
        self.error: str | None = None
        self.params: FrontendParams | None = None
        self.threshold: float | None = None
        self.scorer: WindowScorer | None = None
        self.metrics: dict[str, Any] = {}
        self.export_report: dict[str, Any] = {}
        self.model_card: dict[str, Any] = {}
        self.policy = policy or PolicyConfig()
        self._last_detect_ms: float | None = None

        try:
            self._load(threshold_override)
            self.loaded = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    # -- loading ----------------------------------------------------------
    def _load(self, threshold_override: float | None) -> None:
        frontend_file = self.run_dir / "frontend.json"
        model_file = self.run_dir / "model_int8.tflite"
        if not frontend_file.exists():
            raise FileNotFoundError(f"{frontend_file} not found - run `make train model-export`")
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} not found - run `make model-export`")

        self.params = FrontendParams.from_dict(json.loads(frontend_file.read_text()))
        for name, attr in (("metrics.json", "metrics"), ("export_report.json", "export_report")):
            p = self.run_dir / name
            if p.exists():
                setattr(self, attr, json.loads(p.read_text()))

        # Quantisation parameters: prefer the ones recorded at export time (they
        # are the ones baked into the firmware header), fall back to frontend.json.
        q = self.export_report.get("feature_quantisation") or json.loads(frontend_file.read_text()).get("quant")
        if q:
            self.params.quant = QuantParams.from_dict(q)

        self.threshold = float(threshold_override) if threshold_override is not None else float(
            self.metrics.get("threshold", 0.5)
        )
        self.policy = PolicyConfig(
            threshold=self.threshold,
            confirm_windows=self.policy.confirm_windows,
            refractory_ms=self.policy.refractory_ms,
        )
        self.scorer = WindowScorer(tflite_path=str(model_file), quant=self.params.quant)
        self.model_card = self._build_model_card()

    def _build_model_card(self) -> dict[str, Any]:
        streaming = {}
        p = self.run_dir / "streaming_report.json"
        if p.exists():
            streaming = json.loads(p.read_text())
        return {
            "run_name": self.run_dir.name,
            "keyword": self.metrics.get("keyword", "hi bixby"),
            "threshold": self.threshold,
            "confirm_windows": self.policy.confirm_windows,
            "params": self.metrics.get("params_total"),
            "model_bytes": self.export_report.get("tflite_bytes"),
            "macs_per_window": self.export_report.get("macs_per_window"),
            "integer_only": self.export_report.get("integer_only"),
            "ops": self.export_report.get("ops", []),
            "val_auc": self.metrics.get("val_auc"),
            "val_ap": self.metrics.get("val_ap"),
            "val_recall_at_threshold": self.metrics.get("val_recall_at_threshold"),
            "val_negative_hours": self.metrics.get("val_negative_hours"),
            "int8_auc": self.export_report.get("int8_auc"),
            "float_auc": self.export_report.get("float_auc"),
            "streaming": streaming.get("policies", {}),
            "frontend": {
                "sample_rate": self.params.sample_rate if self.params else None,
                "frame_ms": 30,
                "hop_ms": 20,
                "mel_bins": self.params.num_mel_bins if self.params else None,
                "context_frames": self.params.context_frames if self.params else None,
                "window_ms": (self.params.window_samples / self.params.sample_rate * 1000.0) if self.params else None,
            },
        }

    # -- detection --------------------------------------------------------
    def _to_16k(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            audio = resample(audio, sample_rate, SAMPLE_RATE)
        return audio

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        policy: PolicyConfig | None = None,
        max_windows: int = 6000,
    ) -> dict[str, Any]:
        """Score every 20 ms window of ``audio`` and return the decision timeline.

        Returns per-window scores (so the UI can draw the actual curve the model
        produced), the detections under the requested policy, and timing.
        """
        if not self.loaded or self.params is None or self.scorer is None:
            raise RuntimeError(f"model not loaded: {self.error}")

        x = self._to_16k(audio, sample_rate)
        pol = policy or self.policy
        t0 = time.perf_counter()
        scores = score_waveform_offline(x, self.scorer, self.params)
        self._last_detect_ms = (time.perf_counter() - t0) * 1000.0

        if scores.size > max_windows:  # keep the payload bounded
            scores = scores[:max_windows]
        detections = apply_policy(scores, pol)
        duration_s = x.size / SAMPLE_RATE
        return {
            "duration_s": round(duration_s, 3),
            "hop_ms": 20,
            "threshold": pol.threshold,
            "policy": {
                "confirm_windows": pol.confirm_windows,
                "refractory_ms": pol.refractory_ms,
            },
            "scores": [round(float(s), 4) for s in scores],
            "detections": [
                {
                    "time_s": round(d.time_s, 4),
                    "window_index": d.window_index,
                    "score": round(d.score, 4),
                    "latency_after_window_ms": round(d.time_s * 1000.0, 1),
                }
                for d in detections
            ],
            "detect_ms": round(self._last_detect_ms, 2) if self._last_detect_ms else None,
            "real_time_factor": round(self._last_detect_ms / 1000.0 / duration_s, 4) if duration_s > 0 else None,
            "runtime": "tflite_int8",
        }

    def detect_ulaw(self, payload: bytes, upload_sr: int = 8000, policy: PolicyConfig | None = None) -> dict[str, Any]:
        """Convenience path for the device uplink format (8 kHz G.711 mu-law)."""
        pcm = ulaw_decode(np.frombuffer(payload, dtype=np.uint8))
        return self.detect(float_from_pcm16(pcm), upload_sr, policy=policy)

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "error": self.error,
            "run_dir": str(self.run_dir),
            "model_card": self.model_card,
        }

    @staticmethod
    def window_time_s(window_index: int) -> float:
        return detection_time_s(window_index)
