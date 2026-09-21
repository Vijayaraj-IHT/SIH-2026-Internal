"""Server configuration (paths, ports, tunables) - all overridable by env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass
class Settings:
    """Runtime settings.  Everything has a sane default so `make serve` just works."""

    # -- artifacts --------------------------------------------------------
    artifacts_root: Path = field(default_factory=lambda: _env_path("KWS_ARTIFACTS", REPO_ROOT / "artifacts"))
    run_name: str = field(default_factory=lambda: os.environ.get("KWS_RUN_NAME", "hb-dscnn-w100"))
    data_cache: Path = field(default_factory=lambda: _env_path("KWS_CACHE_DIR", REPO_ROOT / "data" / "cache"))

    # -- storage ----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: _env_path("KWS_DB", REPO_ROOT / "server" / "data" / "telemetry.db"))

    # -- ASR --------------------------------------------------------------
    #: "auto" tries cloud (if a key is present) then offline backends.
    asr_backend: str = field(default_factory=lambda: os.environ.get("KWS_ASR_BACKEND", "auto"))
    asr_uplink_sample_rate: int = 8000  # G.711 mu-law telephony rate
    #: ASR models are trained on 16 kHz; the uplink is upsampled before decoding.
    asr_decode_sample_rate: int = 16000
    asr_timeout_s: float = 20.0

    # -- detector ---------------------------------------------------------
    #: Override the threshold that training selected (useful for live tuning).
    threshold_override: float | None = field(
        default_factory=lambda: float(os.environ["KWS_THRESHOLD"]) if os.environ.get("KWS_THRESHOLD") else None
    )
    confirm_windows: int = field(default_factory=lambda: int(os.environ.get("KWS_CONFIRM", "2")))
    refractory_ms: int = field(default_factory=lambda: int(os.environ.get("KWS_REFRACTORY_MS", "1200")))

    # -- misc -------------------------------------------------------------
    max_upload_mb: float = 24.0
    telemetry_limit: int = 500
    demo_max_seconds: float = 120.0

    @property
    def run_dir(self) -> Path:
        return self.artifacts_root / self.run_name

    @property
    def model_path(self) -> Path:
        return self.run_dir / "model_int8.tflite"


settings = Settings()
