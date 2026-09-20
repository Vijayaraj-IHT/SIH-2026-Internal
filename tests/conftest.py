"""Shared pytest fixtures.

The suite is designed so that ``pytest`` gives useful signal even on a fresh
clone with no datasets: tests that need the corpus are skipped with an explicit
reason instead of failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path.home() / ".cache" / "kws-datasets" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "cache"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_RUN = "hb-dscnn-w100"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "needs_data: requires the downloaded corpora")
    config.addinivalue_line("markers", "needs_cache: requires data/cache TFRecords")
    config.addinivalue_line("markers", "needs_model: requires a trained+exported run in artifacts/")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    if not DATA_DIR.exists():
        pytest.skip(f"corpora not downloaded ({DATA_DIR}); run `make data`")
    return DATA_DIR


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    if not (CACHE_DIR / "cache_index.json").exists():
        pytest.skip(f"feature cache missing ({CACHE_DIR}); run `make cache`")
    return CACHE_DIR


@pytest.fixture(scope="session")
def run_dir() -> Path:
    path = ARTIFACTS_DIR / DEFAULT_RUN
    if not (path / "frontend.json").exists():
        pytest.skip(f"no trained run at {path}; run `make train model-export`")
    return path
