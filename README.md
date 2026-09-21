# Low-Latency Edge Voice Activator for Offline ASR

**SIH 2026 · Problem statement SIH26172**

A device that listens continuously, recognises a custom wake word **entirely on-device**,
and only then streams compressed speech to a recogniser that works without internet.
The point is the combination the problem asks for: always-on listening that costs
almost no CPU and no bandwidth, with speech recognition that still works when the
network does not.

```
"Hi Bixby"  ──▶  [ESP32-S3: int8 CNN, 65 k params]  ──▶  µ-law 64 kbit/s  ──▶  [offline ASR]
                 always listening, never transmits        only after wake
```

## Status

Everything below is implemented and importable, and the test suite passes without
hardware or a network connection. What has **not** happened yet is a training run
on the corrected data pipeline, so this repository currently contains **no valid
accuracy number**:

| Area | State |
|---|---|
| Data pipeline, placement geometry, cache builder | fixed and tested |
| C front-end + G.711 | compiled, and parity-checked against Python: **1960/1960 int8 feature values identical** (`make frontend-check`) |
| µ-law uplink codec | bit-exact against the ITU reference vectors |
| Model, training loop, export, streaming, server | written, unit-tested, not yet exercised end to end |
| ESP32 firmware itself | sources only — no ESP-IDF toolchain on this machine, so it has never been built |
| Trained model with a quotable test score | **pending** — see `artifacts/README.md` |

The three data bugs that blocked training (displaced keyword spans, class-ordered
batches, an evaluation placement that produced zero positives) were fixed after the
runs in `artifacts/` were produced; those runs are kept only as diagnostics.

## What is in this repository

| Path | What it does |
|---|---|
| `ml/data/` | corpus download, manifest building, TFRecord cache, window placement + augmentation |
| `ml/models/dscnn.py` | the DS-CNN (depthwise-separable CNN) in Keras |
| `ml/tools/train.py` | training loop: pos-weighted loss, cosine LR, early stop on val AP, threshold selection |
| `ml/tools/evaluate.py` | held-out test report: AUC/AP, float vs int8, recall at a false-activation budget, per-confusable breakdown |
| `ml/tools/export_tflite.py` | full-integer int8 export + the C array the firmware compiles |
| `ml/tools/streaming_eval.py` | streaming benchmark: latency, false activations per hour, SNR sweep |
| `ml/common/streaming.py` | the sliding-window detector, decision policy and online/offline scoring |
| `edge/common/` | portable C99 log-mel front-end and G.711 µ-law codec (no allocation, no OS); parity with the Python front-end is checked by `tests/test_frontend_parity.py` |
| `edge/host_sim/` | host harness: front-end parity dump, timing bench, µ-law check |
| `edge/esp32/` | ESP-IDF firmware: I2S capture, inference, wake uplink |
| `server/` | FastAPI service: wake events, µ-law ASR endpoint, websocket streaming, dashboard |
| `tests/` | codec, front-end, placement, streaming and server tests (all run without hardware) |

## Quick start

```bash
make install          # virtualenv + dependencies (CPU-only TensorFlow)
make data             # download the corpora (~1 GB, once)
make cache            # build the TFRecord cache
make train            # train the wake-word model          -> artifacts/<run>/
make export           # int8 export + tensor-arena estimate
make frontend         # generate the C tables the firmware needs (after export)
make frontend-check   # build the C front-end + check it against Python (no trained run needed)
make test             # run the test suite
make serve            # dashboard + ASR endpoints at http://localhost:8000
```

Everything after `make data` works with the network unplugged.

## The wake word

The default model is trained on **"Hi Bixby"** — a real, multi-speaker corpus
(`hi-bixby-wakeword-dataset`: 1,200 positives from 3 speakers, 600 phonetically
confusable hard negatives such as "Hi Vicky" / "Hi Dixie", and real urban background
recordings) extended with 12,000 Speech Commands utterances as generic negatives.

Changing the keyword is a data change, not a code change: point
`ml/data/prepare_dataset.py` at another corpus and retrain. The placement, training
and export code is keyword-agnostic.

One caveat, because it cost real debugging time: this corpus has **no
onset/offset annotations** — the format is indexed, but nothing marks where the
word sits inside a 1.7–6.4 s recording. The boundaries are therefore *inferred*
by `ml/data/build_cache.py::speech_bounds`, which anchors on the loudest frame,
grows outward, and caps the span at 1.1 s. That inference is the weakest link in
the pipeline: if it is wrong, windows labelled positive contain silence, and a
model trained on them can still score well on a validation set by exploiting
whatever nuisance cue separates the classes. `tests/test_placement.py` pins the
placement geometry, and `artifacts/README.md` records the incident.

## How the problem's constraints are met

| Requirement | Where it is satisfied |
|---|---|
| Works offline / low bandwidth | `server/asr.py` (local backends first), `edge/` (no network while listening), µ-law 64 kbit/s uplink |
| Latency of wake → transcription | `ml/tools/streaming_eval.py` measures it end to end; the policy's cost is asserted in `tests/test_streaming.py` |
| Low memory | int8 model + tensor arena sized by `edge/tools/size_arena.py`; the firmware logs the real arena usage |
| Low idle CPU | one 512-point FFT + one int8 inference per 20 ms; `edge/esp32/main/bench_task.c` prints the duty cycle |
| Open-source TinyML only | TensorFlow Lite for Microcontrollers, PocketSphinx, NumPy, FastAPI — no proprietary components |

## Honest notes

* The **speaker-disjoint test split is the only number worth quoting.** Validation
  windows come from a speaker that never appears in training; the cache index records
  the split and the manifests record the speaker so this is checkable, not asserted.
* The per-confusable false-activation breakdown in `evaluate.py` exists because an
  aggregate false-activation rate hides which phrase is doing the damage.
* Host-side timings are labelled as host-side. The device numbers come from
  `bench_task.c` on the board; nothing is extrapolated from a laptop.
