/* Dashboard client.
 *
 * One WebSocket at /ws (same origin as this page). Reconnects with capped
 * exponential backoff. Rendering is throttled to ~30 Hz in a rAF loop so a
 * 25-30 Hz feed never queues up more work than the screen can show.
 */
(() => {
"use strict";

const CHANNELS = ["pitot", "gps", "baro", "gyro", "mag", "vane", "accel"];
const CH_LABEL = {
  pitot: "Pitot", gps: "GPS", baro: "Baro",
  gyro: "Gyro", mag: "Mag", vane: "Vane", accel: "Accel",
};
const CH_SUB = {
  pitot: "indicated airspeed", gps: "groundspeed", baro: "altitude",
  gyro: "body rates p/q/r", mag: "heading", vane: "angle of attack",
  accel: "body acceleration",
};
const FAULT_LABEL = {
  pitot_blockage: "Pitot Blockage",
  vane_stuck: "Vane Stuck",
  gyro_drift: "Gyro Drift",
  mag_drift: "Mag Drift",
  baro_bias: "Baro Bias",
  gps_noise: "GPS Noise",
  accel_noise: "Accel Noise",
};

const SCORE_SCALE = 1.0;  // bar full-scale; threshold lives in `state.threshold`
const SPARK_LEN = 75;

const state = {
  ws: null,
  backoff: 500,
  latest: null,        // last telemetry message
  status: { ok: false, message: "connecting…" },
  units: {},
  faults: [],
  threshold: 0.2,
  weights: "",
  activeFault: null,
  spark: Object.fromEntries(CHANNELS.map((c) => [c, []])),
  lastRender: 0,
  sawData: false,
};

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- helpers */
function chip(id, text, cls) {
  const el = $(id);
  el.textContent = text;
  el.className = "chip" + (cls ? " " + cls : "");
}

function fmt(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  return Number(x).toFixed(digits);
}

/* ------------------------------------------------------------ WebSocket */
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let ws;
  try {
    ws = new WebSocket(`${proto}://${location.host}/ws`);
  } catch (e) {
    scheduleReconnect();
    return;
  }
  state.ws = ws;
  chip("chip-conn", "connecting", "");

  ws.onopen = () => {
    state.backoff = 500;
    chip("chip-conn", "live", "chip--ok");
    setStatus(true, "connected to backend");
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    handle(msg);
  };

  ws.onerror = () => { try { ws.close(); } catch {} };

  ws.onclose = () => {
    chip("chip-conn", "reconnecting", "chip--bad");
    setStatus(false, `backend unreachable – retrying in ${(state.backoff / 1000).toFixed(1)}s`);
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  const delay = state.backoff;
  state.backoff = Math.min(state.backoff * 1.7, 5000);
  setTimeout(connect, delay);
}

function send(obj) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(obj));
    return true;
  }
  return false;
}

function handle(msg) {
  switch (msg.type) {
    case "hello":
      state.units = msg.units || {};
      state.faults = msg.faults || [];
      state.threshold = msg.threshold || 0.2;
      state.weights = msg.weights || "";
      chip("chip-weights", shortWeights(state.weights), "chip--warn");
      buildFaultButtons();
      break;

    case "telemetry":
      state.latest = msg;
      state.sawData = true;
      break;

    case "status":
      setStatus(!!msg.ok, msg.message || (msg.ok ? "live" : "no signal"));
      break;

    case "fault_state":
      state.activeFault = msg.active_fault || null;
      // the server reports the *channel*; map back to whichever preset owns it
      syncFaultButtons();
      break;

    case "error":
      setStatus(false, "server error: " + (msg.message || "unknown"));
      break;
  }
}

function shortWeights(w) {
  if (!w) return "weights –";
  return w.toLowerCase().startsWith("random") ? "head UNTRAINED" : "weights trained";
}

/* ------------------------------------------------------------- status UI */
function setStatus(ok, text) {
  state.status = { ok, message: text };
  const strip = $("status-strip");
  strip.className = "status-strip " + (ok ? "status-strip--ok" : "status-strip--bad");
  $("status-text").textContent = text;
}

/* ------------------------------------------------------- fault buttons */
function buildFaultButtons() {
  const box = $("faults");
  box.innerHTML = "";
  for (const f of state.faults) {
    const b = document.createElement("button");
    b.className = "fault-btn";
    b.type = "button";
    b.dataset.fault = f;
    b.textContent = FAULT_LABEL[f] || f;
    b.onclick = () => {
      if (state.activeFault === f) send({ cmd: "clear" });
      else send({ cmd: "set_fault", fault: f });
    };
    box.appendChild(b);
  }
  const clear = document.createElement("button");
  clear.className = "fault-btn fault-btn--clear";
  clear.type = "button";
  clear.dataset.fault = "";
  clear.textContent = "Clear";
  clear.onclick = () => send({ cmd: "clear" });
  box.appendChild(clear);
  syncFaultButtons();
}

/** Highlight the button whose preset matches the active channel. */
function syncFaultButtons() {
  const active = state.activeFault;
  document.querySelectorAll(".fault-btn").forEach((b) => {
    const preset = b.dataset.fault;
    let on = false;
    if (preset) {
      const spec = PRESET_CHANNEL[preset];
      on = !!active && spec === active;
    } else {
      on = !active;  // Clear lights up when nothing is active
    }
    b.classList.toggle("is-active", on);
  });
}

/** preset name -> channel it corrupts (kept in sync with backend faults.py) */
const PRESET_CHANNEL = {
  pitot_blockage: "pitot",
  vane_stuck: "vane",
  gyro_drift: "gyro",
  mag_drift: "mag",
  baro_bias: "baro",
  gps_noise: "gps",
  accel_noise: "accel",
};

/* --------------------------------------------------- channel table rows */
function buildChannelRows() {
  const tbody = document.querySelector("#channels tbody");
  tbody.innerHTML = "";
  for (const c of CHANNELS) {
    const tr = document.createElement("tr");
    tr.id = "row-" + c;
    tr.innerHTML = `
      <td><span class="ch-name">${CH_LABEL[c]}</span>
          <span class="ch-sub">${CH_SUB[c]}</span></td>
      <td class="num" id="v-${c}">–</td>
      <td class="num" id="t-${c}">–</td>
      <td class="num delta" id="d-${c}">–</td>
      <td><div class="score-wrap">
            <span class="score-bar"><span class="score-fill" id="s-${c}"></span></span>
            <span class="score-num" id="n-${c}">0.00</span>
          </div></td>
      <td><canvas class="spark" id="k-${c}" width="150" height="30"></canvas></td>`;
    tbody.appendChild(tr);
  }
}

/* -------------------------------------------------- attitude indicator */
function drawADI(pitch, roll) {
  const c = $("adi");
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, cx = W / 2, cy = H / 2, R = W / 2 - 4;
  const PPD = 5;                 // pixels per degree of pitch
  const rad = (roll * Math.PI) / 180;

  ctx.clearRect(0, 0, W, H);

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2);
  ctx.clip();

  ctx.translate(cx, cy);
  ctx.rotate(-rad);              // world counter-rotates against the aircraft

  const dy = pitch * PPD;        // nose up -> horizon drops on screen
  const B = 3000;

  ctx.fillStyle = "#2f6f9f";     // sky
  ctx.fillRect(-B, -B + dy, 2 * B, B);
  ctx.fillStyle = "#6b4423";     // ground
  ctx.fillRect(-B, dy, 2 * B, B);

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-B, dy);
  ctx.lineTo(B, dy);
  ctx.stroke();

  // pitch ladder
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "#ffffff";
  for (let p = -60; p <= 60; p += 10) {
    if (p === 0) continue;
    const y = dy - p * PPD;
    if (y < -R - 40 || y > R + 40) continue;
    const half = p % 20 === 0 ? 34 : 20;
    ctx.beginPath();
    ctx.moveTo(-half, y);
    ctx.lineTo(half, y);
    ctx.stroke();
    if (p % 20 === 0) {
      const s = String(Math.abs(p));
      ctx.fillText(s, -half - 14, y + 4);
      ctx.fillText(s, half + 14, y + 4);
    }
  }
  ctx.restore();

  // fixed aircraft symbol
  ctx.save();
  ctx.translate(cx, cy);
  ctx.strokeStyle = "#ffd400";
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(-58, 0); ctx.lineTo(-20, 0); ctx.lineTo(-10, 9);
  ctx.moveTo(58, 0);  ctx.lineTo(20, 0);  ctx.lineTo(10, 9);
  ctx.stroke();
  ctx.fillStyle = "#ffd400";
  ctx.beginPath();
  ctx.arc(0, 0, 3.2, 0, Math.PI * 2);
  ctx.fill();

  // roll scale at the top
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1.5;
  for (const a of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
    const t = ((a - 90) * Math.PI) / 180;
    const r1 = R - 12;
    const r2 = R - (a % 30 === 0 ? 24 : 18);
    ctx.beginPath();
    ctx.moveTo(Math.cos(t) * r1, Math.sin(t) * r1);
    ctx.lineTo(Math.cos(t) * r2, Math.sin(t) * r2);
    ctx.stroke();
  }
  // pointer follows the bank angle
  ctx.save();
  ctx.rotate(rad);
  ctx.fillStyle = "#ffd400";
  ctx.beginPath();
  ctx.moveTo(0, -R + 14);
  ctx.lineTo(-6, -R + 26);
  ctx.lineTo(6, -R + 26);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.restore();
}

/* ------------------------------------------------------------ sparkline */
function drawSpark(ch) {
  const c = $("k-" + ch);
  if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  const data = state.spark[ch];
  ctx.clearRect(0, 0, W, H);
  if (data.length < 2) return;

  const thrY = H - (Math.min(state.threshold, SCORE_SCALE) / SCORE_SCALE) * H;
  ctx.strokeStyle = "#3d5673";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, thrY);
  ctx.lineTo(W, thrY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const n = data.length;
  for (let i = 0; i < n; i++) {
    const x = (i / (SPARK_LEN - 1)) * W;
    const v = Math.min(data[i], SCORE_SCALE) / SCORE_SCALE;
    const y = H - v * (H - 2) - 1;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

/* --------------------------------------------------------------- render */
function render(now) {
  requestAnimationFrame(render);
  if (now - state.lastRender < 33) return;   // ~30 Hz
  state.lastRender = now;

  const m = state.latest;
  if (!m) return;

  const a = m.attitude || {};
  drawADI(a.pitch || 0, a.roll || 0);
  $("r-pitch").textContent = fmt(a.pitch) + "°";
  $("r-roll").textContent = fmt(a.roll) + "°";
  $("r-hdg").textContent = fmt(a.hdg) + "°";

  const raw = m.raw || {};
  $("g-ias").textContent = fmt(raw.ias, 1);
  $("g-alt").textContent = fmt(raw.alt, 0);
  $("g-vs").textContent = fmt(raw.vs, 1);
  $("g-mach").textContent = fmt(raw.mach, 3);
  $("g-alpha").textContent = fmt(raw.alpha, 1);
  $("g-thr").textContent = fmt(raw.thr === undefined ? null : raw.thr * 100, 0);

  const st = m.stats || {};
  $("m-ok").textContent = st.frames_ok ?? 0;
  $("m-bad").textContent = st.frames_bad ?? 0;
  $("m-age").textContent = fmt(st.age_s, 2);
  $("m-win").textContent = st.window ?? 0;
  chip("chip-fps", `${st.fps ?? 0} fps`);

  // fault strip
  const ft = $("fault-text");
  ft.textContent = m.active_fault_name
    ? "ACTIVE: " + (FAULT_LABEL[m.active_fault] || m.active_fault_name)
    : "";

  renderVerdict(m);
  renderChannels(m);
}

function renderVerdict(m) {
  const v = m.verdict || {};
  const banner = $("banner");
  const sensor = v.top_sensor || "none";
  const isFault = sensor !== "none";

  banner.className = "banner " + (v.warmup ? "banner--warmup" : isFault ? "banner--fault" : "banner--none");
  $("verdict-sensor").textContent =
    (isFault ? sensor : "none") +
    (v.confidence ? "  " + Math.round(v.confidence * 100) + "%" : "");
  $("verdict-reason").textContent = v.reason || "";

  // encoder (deterministic) vs head (untrained) -- show both, always
  const scores = v.sensor_scores || {};
  let top = null, topVal = -Infinity;
  for (const c of CHANNELS) {
    const s = scores[c] || 0;
    if (s > topVal) { topVal = s; top = c; }
  }
  const open = v.gate_open;
  chip(
    "chip-gate",
    "gate " + (v.warmup ? "—" : open ? "OPEN" : "closed"),
    v.warmup ? "" : open ? "chip--bad" : "chip--ok"
  );
  chip(
    "chip-encoder",
    v.warmup ? "encoder warming" : `encoder ${top} ${topVal.toFixed(2)}`,
    v.warmup ? "" : open ? "chip--bad" : "chip--ok"
  );
}

function renderChannels(m) {
  const ch = m.channels || {};
  const tr = m.truth || {};
  const scores = (m.verdict && m.verdict.sensor_scores) || {};
  const active = m.active_fault;

  for (const c of CHANNELS) {
    const val = ch[c];
    const base = tr[c];
    $("v-" + c).textContent = fmt(val, 3);
    $("t-" + c).textContent = fmt(base, 3);

    const dEl = $("d-" + c);
    const delta = (val !== undefined && base !== undefined) ? val - base : null;
    dEl.textContent = delta === null ? "–" : fmt(delta, 3);
    dEl.classList.toggle("delta--big", delta !== null && Math.abs(delta) > 1e-9);

    const s = scores[c] || 0;
    const fill = $("s-" + c);
    fill.style.width = Math.min(100, (s / SCORE_SCALE) * 100) + "%";
    fill.classList.toggle("is-hot", s >= state.threshold);
    $("n-" + c).textContent = s >= 100 ? s.toExponential(1) : s.toFixed(2);

    const arr = state.spark[c];
    arr.push(s);
    if (arr.length > SPARK_LEN) arr.shift();
    drawSpark(c);

    // highlight the row actually being corrupted
    $("row-" + c).classList.toggle("is-fault", active === c);
  }
}

/* ----------------------------------------------------------------- boot */
buildChannelRows();
drawADI(0, 0);
requestAnimationFrame(render);
connect();

// Surface a plain-language banner if the socket never comes up at all.
setTimeout(() => {
  if (!state.sawData && !state.status.ok) {
    setStatus(false, "no telemetry yet – is server.py running, and is FlightGear up?");
  }
}, 4000);

})();
