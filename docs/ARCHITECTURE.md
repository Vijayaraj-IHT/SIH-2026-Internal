# Architecture

How a spoken word becomes a transcript, and why each hop exists.

```
 ┌──────────────────────────── ESP32-S3 (device, offline-capable) ───────────────────────────┐
 │                                                                                            │
 │  I2S mic ──16 kHz/16-bit──▶ log-mel front-end ──▶ int8 DS-CNN ──▶ decision policy ──▶ LED  │
 │  (DMA, 10 ms blocks)        (30 ms / 20 ms hop)   (49×40 → p)     (τ, 2-of-N, 1.2 s)        │
 │                                     │                                  │                    │
 │                                     └── no network, no allocation ──────┘                    │
 │                                                                        │ on wake               │
 │                                            G.711 µ-law encoder ◀───────┘                     │
 │                                            (16 k → 8 k, 64 kbit/s)                           │
 └────────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                  │  POST /v1/wake      (JSON, ~200 B)
                                                  │  POST /v1/asr/ulaw  (raw µ-law body)
                                                  ▼
 ┌──────────────────────────────── server (reachable, or same box) ──────────────────────────┐
 │  FastAPI ──▶ decode µ-law ──▶ offline ASR (PocketSphinx / Vosk / whisper.cpp) ──▶ response │
 │      │                                                                                     │
 │      └──▶ SQLite telemetry ──▶ /metrics ──▶ dashboard                                      │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Why the split is where it is

**Wake detection runs on the device.** It has to: a system that streams audio to a
server to decide whether anyone spoke burns bandwidth and battery continuously, and
stops working when the link drops. The detector is a 65 k-parameter int8
convolutional network — small enough to sit in a few hundred kilobytes of SRAM and
run every 20 ms alongside everything else.

**Speech recognition runs on the server.** Recognition needs a language model and a
decoder that are far larger than the device's RAM, and its cost is only paid when
someone actually speaks to the device. Keeping it behind an HTTP contract also means
the recogniser is replaceable: the same firmware works with PocketSphinx on a
Raspberry Pi in a village clinic or with a larger model in a district office.

**The transport between them is G.711 µ-law at 8 kHz.** 64 kbit/s instead of
256 kbit/s for raw 16 kHz PCM — a 4× reduction from two independent factors: 8-bit
companding (2×) and the 16 kHz → 8 kHz decimation (2×). G.711 is the codec every
telephone network used, it costs a handful of cycles per sample, and the encoder is
verified bit-exact against the ITU reference in `tests/test_audio.py`.

## The offline story

| Component | Runs offline? | Notes |
|---|---|---|
| Wake detection | yes | entirely on-device; no network access at all |
| ASR | yes, with a local backend | `pocketsphinx` ships in the repo's requirements; `openai` is opt-in only |
| Dashboard | yes | served from the same process; no CDN assets, no external fonts |
| Model training | yes, after `make data` | the corpora download once; everything after is local |
| Telemetry | yes | SQLite file; the dashboard reads it directly |

`server/asr.py` selects a backend in the order `openai → pocketsphinx → reference`.
On a machine with no network and no PocketSphinx model, the `reference` backend still
answers (dynamic-time-warping against a stored template), so the demo never hard-fails
in front of a judge — a deliberate decision, since a demo that depends on a live
network is a demo that fails on stage.

## Latency budget

Measured from the true end of the keyword (not from the start of the utterance):

| Stage | Cost | How it is measured |
|---|---|---|
| Front-end frame (30 ms window, 20 ms hop) | ~20 ms | `edge/host_sim/kws_host_sim --bench`, then on-device via `bench_task.c` |
| Model inference (49×40 int8 DS-CNN) | tens of ms | same |
| Decision policy: N-of-N confirmation | 20 ms per extra window | `apply_policy`, asserted in `tests/test_streaming.py` |
| Uplink: µ-law encode + HTTP | link dependent | `ml/tools/streaming_eval.py`, `/metrics` p50/p90 |

The policy's confirmation window is the one deliberate latency-for-accuracy trade in
the system, and its cost is exactly one hop per extra window — a number the tests
pin down so it cannot silently drift.

## What is deliberately *not* here

* **No cloud dependency.** Nothing in the default path calls an external service.
* **No dynamic allocation on the audio path.** The tensor arena is allocated once at
  boot; the firmware cannot fail later with heap fragmentation.
* **No wake-word-specific ASR.** The device does not try to write the transcript
  itself; it streams audio only after waking, which is what keeps idle power low.
