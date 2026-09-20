/* SIH26172 console - vanilla JS, no build step.
 *
 * Three things it does, all with real data from the server:
 *   1. Stream demo : fetches a reproducible stream whose per-window scores were
 *      produced by the shipped int8 model, animates the playhead, fires a wake
 *      event at the threshold crossing, then uplinks the following seconds as
 *      8 kHz mu-law and asks the server to transcribe them.
 *   2. Mic         : records raw PCM in the browser, posts it to /v1/detect and
 *      draws the returned decision timeline.
 *   3. Fleet       : renders telemetry aggregates from /v1/metrics.
 */

const $ = (sel) => document.querySelector(sel);
const api = (path, opts) => fetch(path, opts);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`#tab-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "fleet") refreshFleet();
    if (tab.dataset.tab === "model") loadModelCard();
  });
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshFleet(); });

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
let CONFIG = null;
async function loadHealth() {
  const h = await (await api("/v1/health")).json();
  const m = $("#pill-model"), a = $("#pill-asr"), d = $("#pill-demo");
  m.textContent = h.model_loaded ? `model ✓ ${h.run_name}` : `model ✗ ${h.model_error || "not loaded"}`;
  m.className = "pill " + (h.model_loaded ? "ok" : "bad");
  a.textContent = `ASR ✓ ${h.asr_backend.name}`;
  a.className = "pill " + (h.asr_backend.available ? "ok" : "bad");
  d.textContent = h.demo_available ? "corpus ✓ cached" : "corpus ✗ missing";
  d.className = "pill " + (h.demo_available ? "ok" : "bad");
  $("#footer-note").textContent = h.demo_available ? "" : h.demo_unavailable_reason || "";
  return h;
}

async function loadConfig() {
  CONFIG = await (await api("/v1/config")).json();
  if (CONFIG.threshold !== null) $("#mic-threshold").value = fmt(CONFIG.threshold, 3);
  $("#mic-confirm").value = CONFIG.confirm_windows || 2;
  return CONFIG;
}

// ---------------------------------------------------------------------------
// Canvas helpers
// ---------------------------------------------------------------------------
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(600, rect.width * dpr);
  canvas.height = canvas.height || 300;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.width / dpr, h: canvas.height / dpr };
}
window.addEventListener("resize", () => {
  if (lastStream) drawStream(lastStream, playIndex, streamDetections);
  if (lastMic) drawMic(lastMic);
});

function drawAxes(ctx, w, h, pad, xLabel, yMaxTicks = 5) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#26304f";
  ctx.lineWidth = 1;
  ctx.font = "11px ui-sans-serif, system-ui";
  ctx.fillStyle = "#94a3c4";
  for (let i = 0; i <= yMaxTicks; i++) {
    const y = pad.t + ((h - pad.t - pad.b) * i) / yMaxTicks;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(fmt(1 - i / yMaxTicks, 2), 6, y + 4);
  }
  ctx.fillText(xLabel, w - pad.r - 60, h - 6);
}

// ---------------------------------------------------------------------------
// 1. Stream demo
// ---------------------------------------------------------------------------
let lastStream = null;
let playIndex = 0;
let streamTimer = null;
let streamDetections = [];
let streamSession = null;

async function startStream() {
  stopStream(true);
  const seed = Number($("#seed").value) || 2026;
  const seconds = Math.min(120, Math.max(5, Number($("#seconds").value) || 45));
  $("#readout-stream").textContent = "building stream (synthesising audio + scoring every 20 ms window) …";
  const res = await api(`/v1/demo/session?seconds=${seconds}&seed=${seed}`);
  if (!res.ok) {
    $("#readout-stream").textContent = "demo unavailable: " + (await res.json()).detail;
    return;
  }
  lastStream = await res.json();
  streamSession = lastStream.session_id;
  streamDetections = lastStream.detections || [];
  $("#kw-name").textContent = `“${lastStream.keyword}”`;
  playIndex = 0;
  renderUplinkStats(lastStream.uplink, null);
  $("#detections-table tbody").innerHTML = "";
  $("#transcript").textContent = "—";

  $("#btn-start").disabled = true;
  $("#btn-stop").disabled = false;

  const hopMs = lastStream.hop_ms;
  streamTimer = setInterval(() => {
    playIndex += 2; // 2 windows per tick => 40 ms of audio per frame
    if (playIndex >= lastStream.scores.length) { stopStream(); return; }
    drawStream(lastStream, playIndex, streamDetections);
    const det = streamDetections.find((d) => d.window_index === playIndex || d.window_index === playIndex - 1);
    if (det && !det._sent) { det._sent = true; onWake(det); }
    const t = (playIndex * hopMs) / 1000;
    $("#readout-stream").textContent =
      `t = ${fmt(t, 1)} s · window #${playIndex} · score ${fmt(lastStream.scores[playIndex], 3)} ` +
      `· τ = ${fmt(lastStream.threshold, 3)} · ${streamDetections.length} detection(s)`;
  }, 40);
}

function stopStream(silent = false) {
  if (streamTimer) clearInterval(streamTimer);
  streamTimer = null;
  $("#btn-start").disabled = false;
  $("#btn-stop").disabled = true;
  if (!silent) $("#readout-stream").textContent += "\nstream finished";
}

function drawStream(s, idx, detections) {
  const canvas = $("#canvas-stream");
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { l: 34, r: 12, t: 10, b: 24 };
  drawAxes(ctx, w, h, pad, "time →");
  const n = s.scores.length;
  if (!n) return;
  const X = (i) => pad.l + ((w - pad.l - pad.r) * i) / n;
  const Y = (v) => h - pad.b - (h - pad.t - pad.b) * Math.min(1, Math.max(0, v));

  // threshold
  ctx.strokeStyle = "#ffb020";
  ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(s.threshold)); ctx.lineTo(w - pad.r, Y(s.threshold)); ctx.stroke();
  ctx.setLineDash([]);

  // ground-truth keyword ends
  ctx.strokeStyle = "#a06bff";
  (s.ground_truth || []).forEach((g) => {
    const x = X((g.t_end_s * 1000) / s.hop_ms);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
  });

  // score curve
  ctx.strokeStyle = "#4c8dff";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = X(i), y = Y(s.scores[i]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();

  // detections so far
  ctx.fillStyle = "#21c29a";
  detections.forEach((d) => {
    if (d.window_index > idx) return;
    const x = X(d.window_index);
    ctx.beginPath(); ctx.arc(x, Y(d.score), 4.5, 0, Math.PI * 2); ctx.fill();
  });

  // playhead
  ctx.strokeStyle = "rgba(255,255,255,.35)";
  ctx.beginPath(); ctx.moveTo(X(idx), pad.t); ctx.lineTo(X(idx), h - pad.b); ctx.stroke();
}

async function onWake(det) {
  // 1) the device announces the wake event
  const wakeRes = await (await api("/v1/wake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device_id: "sim-dashboard",
      keyword: lastStream.keyword,
      score: det.score,
      threshold: lastStream.threshold,
      policy: `confirm_${lastStream.policy.confirm_windows}`,
      firmware: "host-sim",
      meta: { source: "stream-demo", session: lastStream.session_id, window: det.window_index },
    }),
  })).json();

  // 2) latency against ground truth (the browser knows both timestamps)
  const truths = (lastStream.ground_truth || []).map((g) => g.t_end_s);
  let nearest = null, best = 1e9;
  truths.forEach((t) => { const d = Math.abs(t - det.time_s); if (d < best) { best = d; nearest = t; } });
  const latencyMs = nearest !== null && best < 2 ? (det.time_s - nearest) * 1000 : null;

  // 3) uplink the following seconds as mu-law and transcribe
  const from = det.time_s;
  const to = Math.min(lastStream.duration_s, det.time_s + lastStream.uplink.window_s);
  const bytes = await (await api(`/v1/demo/uplink?session_id=${lastStream.session_id}&start_s=${from}&end_s=${to}`)).arrayBuffer();
  const asr = await (await api(`/v1/asr/ulaw?device_id=sim-dashboard`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: bytes,
  })).json();

  const truth = nearest !== null ? (lastStream.ground_truth.find((g) => g.t_end_s === nearest) || {}).detail : "";
  addDetectionRow(det, latencyMs, truth, bytes.byteLength, asr);
  $("#transcript").textContent = asr.text || "(no transcript)";
  $("#readout-asr").textContent =
    `${bytes.byteLength} B µ-law → ${asr.backend} · decode ${fmt(asr.decode_ms, 0)} ms · RTF ${fmt(asr.real_time_factor, 3)}` +
    (asr.reduction_ratio !== null ? ` · ${fmt(asr.reduction_ratio * 100, 1)} % smaller than PCM16` : "");
  renderUplinkStats(lastStream.uplink, asr);
  refreshFleet();
}

function addDetectionRow(det, latencyMs, truth, bytes, asr) {
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td>${$("#detections-table tbody").children.length + 1}</td>` +
    `<td>${fmt(det.time_s, 2)} s</td>` +
    `<td>${fmt(det.score, 3)}</td>` +
    `<td>${latencyMs === null ? "—" : fmt(latencyMs, 0) + " ms"}</td>` +
    `<td class="muted">${truth || "—"}</td>` +
    `<td>${bytes} B</td>` +
    `<td>${(asr.text || "—").replace(/</g, "&lt;")}</td>`;
  $("#detections-table tbody").prepend(tr);
}

function renderUplinkStats(uplink, asr) {
  $("#uplink-stats").innerHTML = [
    stat("codec", "G.711 µ-law"),
    stat("uplink rate", `${uplink.ulaw_8k_bits_per_second / 1000} kbit/s`, `PCM16 = ${uplink.pcm16_16k_bits_per_second / 1000} kbit/s`),
    stat("reduction", `${fmt(uplink.reduction_ratio * 100, 0)} %`, `${uplink.bytes_total} B vs ${uplink.bytes_if_pcm16} B`),
    stat("transcript backend", asr ? asr.backend : "—", asr ? `RTF ${fmt(asr.real_time_factor, 3)}` : "waiting for a wake event"),
  ].join("");
}
const stat = (k, v, sub = "") => `<div class="stat"><div class="k">${k}</div><div class="v">${v}${sub ? ` <small>${sub}</small>` : ""}</div></div>`;

// ---------------------------------------------------------------------------
// 2. Microphone
// ---------------------------------------------------------------------------
let mediaRecorder = null;
let recordedChunks = [];
let lastMic = null;

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => e.data.size && recordedChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      await analyseRecording(new Blob(recordedChunks, { type: recordedChunks[0]?.type || "audio/webm" }));
    };
    mediaRecorder.start();
    $("#btn-rec").disabled = true;
    $("#btn-rec-stop").disabled = false;
    $("#mic-status").textContent = "recording … say “hi bixby”, then stop";
  } catch (err) {
    $("#mic-status").textContent = "microphone unavailable: " + err.message + " (https or localhost required)";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  $("#btn-rec").disabled = false;
  $("#btn-rec-stop").disabled = true;
  $("#mic-status").textContent = "analysing …";
}

async function analyseRecording(blob) {
  // The recorder gives us webm/ogg; decode it in the browser to raw PCM so the
  // server receives exactly what the device microphone would produce.
  let pcm16, sampleRate;
  try {
    const arrayBuf = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await ctx.decodeAudioData(arrayBuf);
    sampleRate = decoded.sampleRate;
    const ch = decoded.getChannelData(0);
    pcm16 = new Int16Array(ch.length);
    for (let i = 0; i < ch.length; i++) pcm16[i] = Math.max(-1, Math.min(1, ch[i])) * 32767;
    await ctx.close();
    $("#mic-status").textContent = `captured ${fmt(decoded.duration, 2)} s @ ${sampleRate} Hz — running the int8 detector`;
  } catch (err) {
    $("#mic-status").textContent = "could not decode recording: " + err.message;
    return;
  }

  const threshold = Number($("#mic-threshold").value);
  const confirm = Number($("#mic-confirm").value) || 2;

  const fd = new FormData();
  fd.append("file", new Blob([pcm16.buffer], { type: "application/octet-stream" }), "mic.pcm");
  const res = await api(`/v1/detect?sample_rate=${sampleRate}&confirm_windows=${confirm}`, { method: "POST", body: fd });
  if (!res.ok) { $("#mic-status").textContent = "detect failed: " + (await res.json()).detail; return; }
  const data = await res.json();
  if (Number.isFinite(threshold)) data.threshold = threshold;
  lastMic = data;
  drawMic(data);

  const nDet = data.detections.length;
  $("#readout-mic").textContent =
    `${fmt(data.duration_s, 2)} s · ${data.scores.length} windows · peak score ${fmt(Math.max(...data.scores), 3)} ` +
    `· τ = ${fmt(data.threshold, 3)} · ${nDet} detection(s) · level ${data.level_dbfs} dBFS` +
    `\ndetect ran in ${fmt(data.detect_ms, 0)} ms (RTF ${fmt(data.real_time_factor, 4)}) on the server's copy of the int8 model`;

  // Send the post-detection window through the real mu-law ASR path.
  if (nDet > 0) {
    const det = data.detections[0];
    const sr = sampleRate;
    const from = Math.floor(det.time_s * sr);
    const to = Math.min(pcm16.length, Math.floor((det.time_s + 4) * sr));
    const slice = pcm16.slice(from, to);
    const ulaw = encodeUlaw(slice); // browser-side G.711 encode, exactly like the firmware
    const asr = await (await api("/v1/asr/ulaw?device_id=browser-mic&sample_rate=" + sr, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: ulaw,
    })).json();
    $("#mic-transcript").textContent = asr.text || "(no transcript)";
    $("#readout-mic").textContent +=
      `\nuplink ${ulaw.length} B µ-law (vs ${slice.length * 2} B PCM16) → ${asr.text ? "“" + asr.text + "”" : "no transcript"}` +
      ` (${asr.backend}, decode ${fmt(asr.decode_ms, 0)} ms)`;
    await api("/v1/wake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_id: "browser-mic", keyword: "hi bixby", score: det.score, threshold: data.threshold,
        latency_ms: null, policy: `confirm_${confirm}`, firmware: "browser",
        meta: { source: "mic", duration_s: data.duration_s },
      }),
    });
    refreshFleet();
  } else {
    $("#mic-transcript").textContent = "—";
  }
  $("#mic-status").textContent = "done — record again to retry";
}

function drawMic(data) {
  const canvas = $("#canvas-mic");
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { l: 34, r: 12, t: 10, b: 24 };
  drawAxes(ctx, w, h, pad, "time →");
  const n = data.scores.length;
  if (!n) return;
  const X = (i) => pad.l + ((w - pad.l - pad.r) * i) / n;
  const Y = (v) => h - pad.b - (h - pad.t - pad.b) * Math.min(1, Math.max(0, v));
  ctx.strokeStyle = "#ffb020"; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(data.threshold)); ctx.lineTo(w - pad.r, Y(data.threshold)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#4c8dff"; ctx.lineWidth = 1.6; ctx.beginPath();
  for (let i = 0; i < n; i++) { const x = X(i), y = Y(data.scores[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.stroke();
  ctx.fillStyle = "#21c29a";
  data.detections.forEach((d) => { ctx.beginPath(); ctx.arc(X(d.window_index), Y(d.score), 5, 0, Math.PI * 2); ctx.fill(); });
}

// G.711 mu-law encoder (the same companding law as the firmware's G711_ulaw_encode).
function encodeUlaw(pcm16) {
  const out = new Uint8Array(pcm16.length);
  const BIAS = 0x84, CLIP = 32635;
  for (let i = 0; i < pcm16.length; i++) {
    let s = pcm16[i];
    const sign = s < 0 ? 0x80 : 0;
    if (s < 0) s = -s;
    if (s > CLIP) s = CLIP;
    s += BIAS;
    let exp = 7;
    for (let mask = 0x4000; exp > 0 && (s & mask) === 0; mask >>= 1) exp--;
    const mant = (s >> (exp + 3)) & 0x0f;
    out[i] = ~(sign | (exp << 4) | mant) & 0xff;
  }
  return out;
}

// ---------------------------------------------------------------------------
// 3. Model card / budget
// ---------------------------------------------------------------------------
async function loadModelCard() {
  const info = await (await api("/v1/model")).json();
  const c = info.model_card || {};
  const rows = [
    ["run", c.run_name], ["keyword", c.keyword], ["parameters", c.params?.toLocaleString()],
    ["int8 model size", c.model_bytes ? `${(c.model_bytes / 1024).toFixed(1)} KiB` : "—"],
    ["MACs / window", c.macs_per_window?.toLocaleString()],
    ["graph", c.integer_only ? "integer-only ✓" : "contains float ✗"],
    ["val AUC", fmt(c.val_auc, 5)], ["val AP", fmt(c.val_ap, 5)],
    ["recall @ operating τ", fmt(c.val_recall_at_threshold * 100, 1) + " %"],
    ["float → int8 AUC", `${fmt(c.float_auc, 4)} → ${fmt(c.int8_auc, 4)}`],
    ["threshold τ", fmt(c.threshold, 4)], ["confirm policy", `${c.confirm_windows} window(s)`],
    ["front-end", `${c.frontend?.sample_rate / 1000} kHz · ${c.frontend?.mel_bins} mel · ${c.frontend?.hop_ms} ms hop · ${c.frontend?.context_frames} frames`],
  ];
  $("#model-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v ?? "—"}</td></tr>`).join("");

  $("#ops-list").innerHTML = (c.ops || []).map((o) => `<li>${o}</li>`).join("");
  $("#pipeline-list").innerHTML = [
    `16 kHz mono capture → 30 ms window / 20 ms hop (${c.frontend?.hop_ms} ms decision cadence)`,
    `40-band log-mel, ${c.frontend?.context_frames}-frame context (${fmt((c.frontend?.window_ms || 0) / 1000, 2)} s)`,
    `int8 DS-CNN, ${c.params?.toLocaleString()} parameters, ${c.macs_per_window?.toLocaleString()} MACs per window`,
    `sigmoid gate at τ = ${fmt(c.threshold, 3)} with ${c.confirm_windows}-window confirmation`,
    `on wake → 8 kHz G.711 µ-law uplink (64 kbit/s) → remote ASR`,
  ].map((t) => `<li>${t}</li>`).join("");

  // Budget bars: what the problem statement constrains us by.
  const budgets = [
    { label: "RAM footprint (limit 256 KB)", value: null, note: "measured on the ESP32-S3 build in docs/BENCHMARKS.md" },
    { label: "CPU while idle-listening (limit 10 %)", value: null, note: "per-window inference cost × decision cadence" },
  ];
  $("#budget").innerHTML = budgets.map((b) => `
    <div class="bar">
      <div class="top"><span>${b.label}</span><span class="muted">see benchmarks</span></div>
      <div class="track"><div class="fill" style="width:100%"></div></div>
      <div class="note">${b.note}</div>
    </div>`).join("");

  const streaming = c.streaming || {};
  const keys = Object.keys(streaming).sort();
  if (keys.length) {
    $("#tradeoff-table").innerHTML =
      "<thead><tr><th>policy</th><th>median latency</th><th>p90 latency</th><th>recall</th><th>FA/h confusables</th><th>FA/h generic</th><th>FA/h background</th></tr></thead><tbody>" +
      keys.map((k) => {
        const p = streaming[k], kw = p.keywords || {};
        return `<tr><td>${k}</td><td>${fmt(kw.median_latency_ms, 0)} ms</td><td>${fmt(kw.p90_latency_ms, 0)} ms</td>
                <td>${fmt((kw.recall || 0) * 100, 1)} %</td>
                <td>${fmt(p.confusables_fa_per_hour, 2)}</td><td>${fmt(p.generic_speech_fa_per_hour, 2)}</td>
                <td>${fmt(p.background_fa_per_hour, 2)}</td></tr>`;
      }).join("") + "</tbody>";
  }
}

// ---------------------------------------------------------------------------
// 4. Fleet telemetry
// ---------------------------------------------------------------------------
async function refreshFleet() {
  const [m, t] = await Promise.all([(await api("/v1/metrics")).json(), (await api("/v1/telemetry?limit=25")).json()]);
  const s = m.uplink_savings;
  $("#fleet-stats").innerHTML = [
    stat("wake events", m.wake_events, `${m.devices} device(s)`),
    stat("wake latency", m.latency_ms.p50 === null ? "—" : `${fmt(m.latency_ms.p50, 0)} ms`, `p90 ${fmt(m.latency_ms.p90, 0)} ms · max ${fmt(m.latency_ms.max, 0)} ms`),
    stat("uplink sent", `${(s.bytes_sent / 1024).toFixed(1)} KB`, `PCM16 would be ${(s.bytes_if_pcm16 / 1024).toFixed(1)} KB`),
    stat("bandwidth saved", s.reduction_ratio === null ? "—" : `${fmt(s.reduction_ratio * 100, 1)} %`, `${(s.bytes_saved / 1024).toFixed(1)} KB not transmitted`),
    stat("ASR requests", m.asr.requests, `${fmt(m.asr.audio_seconds, 1)} s decoded`),
    stat("decode speed", m.asr.avg_rtf === null ? "—" : `RTF ${fmt(m.asr.avg_rtf, 3)}`, `avg ${fmt(m.asr.avg_decode_ms, 0)} ms`),
  ].join("");

  $("#wake-table tbody").innerHTML = t.wake_events.map((e) =>
    `<tr><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td><td>${e.device_id}</td><td>${e.keyword || "—"}</td>
     <td>${fmt(e.score, 3)}</td><td>${fmt(e.threshold, 3)}</td>
     <td>${e.latency_ms === null ? "—" : fmt(e.latency_ms, 0) + " ms"}</td><td>${e.policy || "—"}</td></tr>`).join("");

  $("#asr-table tbody").innerHTML = t.asr_requests.map((e) =>
    `<tr><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td><td>${e.device_id || "—"}</td><td>${e.codec}</td>
     <td>${e.bytes_in}</td><td>${fmt(e.audio_seconds, 2)} s</td><td>${fmt(e.decode_ms, 0)} ms</td>
     <td>${fmt(e.rtf, 3)}</td><td>${(e.text || "—").replace(/</g, "&lt;")}</td></tr>`).join("");

  drawLatencyHistogram(t.wake_events.map((e) => e.latency_ms).filter((x) => x !== null));
}

function drawLatencyHistogram(values) {
  const canvas = $("#canvas-latency");
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { l: 34, r: 12, t: 10, b: 24 };
  drawAxes(ctx, w, h, pad, "ms");
  if (!values.length) {
    ctx.fillStyle = "#94a3c4";
    ctx.fillText("no latency samples yet — run the stream demo or record from the mic", pad.l + 10, h / 2);
    return;
  }
  const max = Math.max(...values, 100);
  const bins = 12;
  const counts = new Array(bins).fill(0);
  values.forEach((v) => counts[Math.min(bins - 1, Math.floor((v / max) * bins))]++);
  const peak = Math.max(...counts);
  const bw = (w - pad.l - pad.r) / bins;
  counts.forEach((c, i) => {
    const bh = ((h - pad.t - pad.b) * c) / peak;
    ctx.fillStyle = "#4c8dff";
    ctx.fillRect(pad.l + i * bw + 2, h - pad.b - bh, bw - 4, bh);
  });
  ctx.fillStyle = "#94a3c4";
  ctx.fillText(`n=${values.length}`, w - pad.r - 40, h - 6);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
$("#btn-start").addEventListener("click", startStream);
$("#btn-stop").addEventListener("click", () => stopStream());
$("#btn-rec").addEventListener("click", startRecording);
$("#btn-rec-stop").addEventListener("click", stopRecording);
$("#btn-reset").addEventListener("click", async () => {
  await api("/v1/telemetry/reset", { method: "POST" });
  refreshFleet();
});

(async function init() {
  await loadHealth();
  await loadConfig();
  await refreshFleet();
  await loadModelCard();
  setInterval(loadHealth, 30000);
})();
