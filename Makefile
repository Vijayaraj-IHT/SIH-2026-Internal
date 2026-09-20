# ---------------------------------------------------------------------------
# SIH26172 - low-latency edge voice activator
#
#   make install     create the virtualenv and install dependencies
#   make data        download the corpora (network)
#   make cache       build the TFRecord cache from the corpora
#   make train       train the wake-word model
#   make export      quantise to int8 + write the C model header
#   make frontend    generate the C front-end tables (needs the export)
#   make host        build + run the host front-end parity/simulation tool
#   make test        run the test suite
#   make bench       latency / false-activation benchmark (needs the export)
#   make evaluate    held-out test-set report (needs the export)
#   make esp32       build the ESP32-S3 firmware (needs export + frontend)
#   make serve       run the ASR server + dashboard at http://localhost:8000
#   make all         data -> cache -> train -> export -> frontend -> test
#
# A full first-time run is network-heavy (a few hundred MB of audio); after that
# `make train export frontend` works entirely offline.
# ---------------------------------------------------------------------------

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUN ?= hb-dscnn-w100
TRAIN_ARGS ?= --epochs 45 --batch-size 128 --width 1.0 --threads 2
CACHE_DIR ?= data/cache
DATA_DIR ?= data/datasets
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help install data cache train export frontend host frontend-check test bench evaluate esp32 serve all clean clean-runs

help:
	@grep -E '^#   make' -m 40 Makefile | sed 's/^#   //'

# ---------------------------------------------------------------------------
.venv:
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip

install: .venv
	$(PIP) install -r requirements-ml.txt
	$(PIP) install -r requirements-server.txt
	$(PIP) install -r requirements-dev.txt
	@echo "installed. TensorFlow is CPU-only on purpose: the whole pipeline must"
	@echo "run on a laptop with no GPU, which is also what the field kit looks like."

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data:
	$(PY) -m ml.data.fetch_datasets --out "$(DATA_DIR)"
	$(PY) -m ml.data.prepare_dataset --raw "$(DATA_DIR)/raw" --out ml/data/manifests

cache: data
	$(PY) -m ml.data.build_cache --manifests ml/data/manifests --out "$(CACHE_DIR)"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
train:
	$(PY) -m ml.tools.train --run-name "$(RUN)" $(TRAIN_ARGS)

export:
	$(PY) -m ml.tools.export_tflite --run-name "$(RUN)"
	$(PY) -m edge.tools.size_arena --run-name "$(RUN)"

frontend: export
	$(PY) -m edge.tools.export_frontend_header --run-name "$(RUN)"

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
test:
	$(PY) -m pytest tests -q

host:
	$(MAKE) -C edge/host_sim

# The window and mel filterbank tables are pure functions of the front-end
# parameters, so the C front-end can be built and checked against the Python
# implementation without a trained model. This is the fast way to catch a
# front-end regression; `make frontend` is the calibrated, deployable path.
frontend-check:
	$(PY) -m edge.tools.export_frontend_header --allow-default-quant
	$(MAKE) -C edge/host_sim
	$(PY) -m pytest tests/test_frontend_parity.py -q

bench: export
	$(PY) -m ml.tools.streaming_eval --run-name "$(RUN)"

evaluate: export
	$(PY) -m ml.tools.evaluate --run-name "$(RUN)"

# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------
esp32:
	cd edge/esp32 && idf.py set-target esp32s3 && idf.py build

serve:
	$(PY) -m uvicorn server.app:app --host $(HOST) --port $(PORT)

# ---------------------------------------------------------------------------
all: cache train export frontend test
	@echo
	@echo "artifacts/$(RUN) is ready. Next: make bench evaluate serve"

clean:
	rm -rf edge/host_sim/gen edge/host_sim/kws_host_sim edge/build
	find . -name '__pycache__' -prune -exec rm -rf {} +

clean-runs:
	rm -rf artifacts/smoke* artifacts/sweep* artifacts/cap* artifacts/w100-lr*
