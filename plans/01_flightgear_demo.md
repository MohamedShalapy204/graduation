# Plan 01 — FlightGear Sensor-Fault Detection Demo

**Status:** ACTIVE
**Owner:** Mohamed (AI + Backend)
**Purpose:** A doctor-ready demo, built on top of FlightGear, ahead of the full system.
**Relationship to ground truth:** This is a vertical slice of §2.3 (Worker Container AI
Fault Detector) and §4 (AI Fault Detection Model) of
`docs/new/system_architecture_implementation_plan(1).md`. It is **not** a replacement for it.

---

## 0. Constraints (from user, non-negotiable)

| # | Constraint |
|---|---|
| C1 | FlightGear handles the **full simulation, physics, and visualization**. We build only on top of it. Never modify FG internals. |
| C2 | The model is written in **Python**, using `torch` (CPU). |
| C3 | **No training now.** Demo only. |
| C4 | The model takes **sensor data directly from FlightGear** and predicts **which sensor has errors**. |
| C5 | Model output must appear in **HTML/JS**. |
| C6 | **One fault at a time.** |

---

## 1. Architecture

```
┌────────────────────────────────────────────────────────────┐
│ FlightGear 2024.1.7 (AppImage)  — physics + 3D + viz      │
│   launched with  --generic=file,out,<rate>,<xml>,telemetry │
│   NO internal modification. FG remains the truth source.   │
└───────────────────────────┬────────────────────────────────┘
                            │ UDP  localhost:5502
                            │ one JSON line per frame (~30 Hz)
                            ▼
┌────────────────────────────────────────────────────────────┐
│ backend/fg_reader.py                                       │
│   socket → parse JSON line → validated state dict          │
└───────────────────────────┬────────────────────────────────┘
                            │ raw truth state
                            ▼
┌────────────────────────────────────────────────────────────┐
│ backend/sensor_channels.py                                 │
│   truth → redundant, overlapping *sensor* channels:        │
│     pitot airspeed, GPS groundspeed, baro alt, gyro rate,  │
│     mag heading, AoA vanes, ...                            │
│   each channel = what a real instrument would report       │
└───────────────────────────┬────────────────────────────────┘
                            │ measurements
                            ▼
┌────────────────────────────────────────────────────────────┐
│ backend/faults.py    — injects ONE fault into ONE channel  │
│   (pitot freeze | gyro drift | baro bias | actuator jam)   │
└───────────────────────────┬────────────────────────────────┘
                            │ possibly-corrupted measurement set
                            ▼
┌────────────────────────────────────────────────────────────┐
│ backend/model.py    — torch.nn.Module  (NO training)       │
│   input:  window of last N sensor readings                 │
│   output: per-sensor fault probability + reason code       │
│   interface is identical to the future trained model       │
└───────────────────────────┬────────────────────────────────┘
                            │ verdict + telemetry
                            ▼
┌────────────────────────────────────────────────────────────┐
│ backend/server.py  — WebSocket + static file server        │
└───────────────────────────┬────────────────────────────────┘
                            │ ws://localhost:8765
                            ▼
┌────────────────────────────────────────────────────────────┐
│ web/index.html + app.js + style.css                        │
│   attitude indicator · gauges · live telemetry table       │
│   fault-injection buttons · model verdict banner           │
└────────────────────────────────────────────────────────────┘
```

### Why this design

**The detection problem.** To claim *"sensor X is wrong"* you need something to compare
against. We solve this with **analytical redundancy**: FlightGear's ground truth is
transformed into several *independent instrument channels that measure the same physical
quantity*. When the channels disagree, the odd one out is the failed sensor.

Example — speed:

| Channel | Source | Corrupted by |
|---|---|---|
| `pitot_airspeed` | /velocities/airspeed-kt | blockage → freezes |
| `gps_groundspeed` | lat/lon rate | unaffected |
| `baro_climb_rate` | altitude rate | unaffected |

If pitot freezes while GPS and baro still show motion → **pitot blockage**.

**Why no training is acceptable here.** The detector is an *analytically-derived*
consistency model implemented as a real `torch.nn.Module`, with weights derived from
physics (redundancy relations), not learned. This is a legitimate, explainable method —
not a random-weight fake.

**Why it is not throwaway.** `model.py` exposes a fixed interface:

```python
class FaultDetector(nn.Module):
    def forward(self, window: Tensor) -> DetectorOutput  # probs, top_sensor, reason
```

When Phase 2 of the ground-truth plan (train LSTM-AE / TCN / Isolation Forest) happens,
we swap the class body. Everything upstream (FG link, channels, faults, WebSocket
contract, dashboard) is untouched.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Autoencoder with random weights | Reconstruction error would be meaningless. Not a model. |
| Aircraft-property injection via Nasal | Modifies FG internals → violates C1. |
| Native FDM socket (`--native-fdm`) | Struct is limited; no instrument-level properties (mag heading, GPS vs baro separation). Generic JSON protocol gives arbitrary property access. |
| Train now | Violates C3. |

---

## 2. Sensor channels to build

| id | Channel | FG property | Redundancy partner | Fault model |
|---|---|---|---|---|
| `pitot` | Indicated airspeed | `/velocities/airspeed-kt` | gps_groundspeed, climb_rate | **freeze** (stuck value) |
| `gps` | Ground speed / position | `/position/latitude-deg`, `/longitude-deg` | pitot + climb_rate | **spike / dropout** |
| `baro` | Altitude | `/position/altitude-ft` | gps altitude, vertical-speed | **bias** (constant offset) |
| `gyro` | Body rates p,q,r | `/velocities/p-body_rad_sec` etc. | finite-diff of Euler angles | **drift** (ramp) |
| `mag` | Magnetic heading | `/orientation/heading-deg` | integrated yaw rate | **drift** (ramp) |
| `vane` | Angle of attack | `/orientation/alpha-deg` | `atan2(w, u)` from body velocity | **stuck / offset** |
| `accel` | Body acceleration | `/accelerations/...` | finite-diff of velocity | **noise / bias** |

Sensors are chosen so each has **at least one independent witness.**
Final set may be trimmed during Phase 1–2 based on what FG actually exposes.

---

## 3. Phases

### Phase 0 — Setup ✅ DONE (verified)
- [x] Delete agent-generated `squashfs-root/`
- [x] Create `Agent.md`
- [x] Create `plans/` with this document
- [x] Copy `rl-project/.venv` → `graduation/.venv` (5.9 G, `cp -a`)
- [x] Repair 22 stale absolute paths left by the copy
      (`bin/*` shebangs, `activate`, `pyvenv.cfg`) — all now point at
      `/home/mohamed/Documents/graduation/.venv`; **zero** refs to rl-project remain
- [x] `pip install websockets` inside `graduation/.venv` **only** → `17.4`
- [x] Verify `import torch` → `2.15.0+cu132`, `cuda.is_available() == False`
- [x] Verify `import websockets`

**Exit (verified):**
```bash
$ source .venv/bin/activate
$ python -c "import torch, websockets; print('ok')"
torch       2.15.0+cu132
cuda        False -> using CPU
websockets  17.4
numpy       2.5.4
ok
```
Note: use `python -m pip`, not `./.venv/bin/pip`, until the venv is rebuilt —
module resolution is correct either way, shebangs are now fixed too.

---

### Phase 1 — FlightGear → Python telemetry link ✅ DONE (verified)

- [x] `fg-fault-demo/protocol/telemetry.xml` — 26 chunks, ASCII, `var_separator=`,``,
      `line_separator=newline`, one `k=v` line per frame. Symlinked into
      `$FG_ROOT/Protocol/telemetry.xml` (that dir is the only lookup path).
- [x] **Transport decided: UDP socket** (`--generic=socket,out,30,localhost,5510,udp,telemetry`).
      `file,out` was used only as the first-launch syntax check, per the risk table.
- [x] `backend/fg_reader.py` — binds UDP 5510, `parse_line()` validates + coerces,
      drops malformed frames (never propagates them), exposes `latest_state()`,
      `age_seconds()`, `states()` generator, `stats`. Includes a `__main__` smoke test.
- [x] Live check against a running FG session.

**Exit criteria — all met**
- ✅ FG launches with the custom `--generic=` flag, no protocol errors in the log
- ✅ `frames ok=270 bad=0 (~22.5 Hz)` — meets the ≥20 Hz acceptance (capped by `--max-fps=30`)
- ✅ Values coherent: `alt=146.06 ft`, `ias=9.06 kt`, `hdg=180.01`, `pitch=4.89`,
      `az=-32.1 ft/s²` (exactly −1 g → parked, physics sane), `t` advancing 274→285 s
- ✅ **Cross-checked against ground truth:** `lat=63.9918, lon=-22.6054` = Keflavík (BIKF),
      published ARP 63.9850°N / 22.6056°W — longitude matches to 0.0002°.
      Confirms property paths read real data, not placeholders.

**Verified property paths (corrected from first guesses)**
| Purpose | Correct path | First guess (WRONG) |
|---|---|---|
| body rates p/q/r | `/orientation/{p,q,r}-body` | `/velocities/p-body` |
| GPS groundspeed witness | `/velocities/speed-{north,east}-fps` | `/velocities/speed-ground-fps` (does not exist) |
| accel body axes | `/accelerations/pilot/{x,y,z}-accel-fps_sec` | `/accelerations/N{x,y,z}` |
| baro | `/position/altitude-ft` + `/instrumentation/altimeter/indicated-altitude-ft` | — |

**Gotchas hit (for future reference)**
- **XML comments must not contain `--`** — a comment containing the literal `--generic`
  made the file unparseable (`ElementTree.ParseError` at line 7). Use `[generic option]`.
- **`<type>` defaults to `int`** in the generic protocol; every float chunk must declare
  `<type>float</type>` or values truncate.
- **FG is a ~2 GB process.** On this 7.4 GB box (Firefox ~2 GB + 3 GB swap in use) the
  kernel OOM-killed it twice. Launch lean: `--ai-models=0 --ai-traffic=0
  --terrasync=0 --sound=0 --splash-screen=0 --random-{vegetation,objects,buildings}=0
  --lod-levels=2`. The AI subsystem was loading Nimitz/Eisenhower carrier models —
  pure waste for this demo.
- `alt=-9999.00` and `t=0.000` for the first ~2 s are **startup transients**, not a bug.
- Don't `rm -f` FG's output file while FG holds it open — it keeps writing to an
  unlinked inode and the path reads 0 bytes.

**Risks (as planned)**
| Risk | Outcome |
|---|---|
| `--generic` flag syntax wrong → FG exits | Cleared: syntax confirmed from `$FG_ROOT/Docs/README.IO` lines 22–34 |
| Property names differ in 2024.1.7 | Cleared: verified against `Protocol/aura-imu.xml` + `flightrecorder/components/position.xml` |
| Frames arrive but are garbage | 0 bad frames out of 270 |
| Rate lower than expected | 22.5 Hz ≥ 20 Hz acceptance |

---

### Phase 2 — Sensor channels + fault injection ✅ DONE (verified)
- [x] `sensor_channels.py`: truth state → channel dict (with units normalised)
- [x] `faults.py`: injectors, **exactly one active at a time**
  - `freeze(channel, at_value)` — pitot blockage
  - `ramp(channel, rate)` — gyro / mag drift
  - `bias(channel, delta)` — baro error
  - `noise(channel, sigma)` — sensor degradation
- [x] `FaultController` with `set_active(fault_type)` / `clear()`
- [x] Log: clean channels vs corrupted channel, side by side

**Exit criteria:** toggling a fault in code visibly corrupts exactly one channel and no other.

**Verified against live FlightGear (all 7 presets):**
| Fault | Channel | Result |
|---|---|---|
| `pitot_blockage` | pitot | ✅ (delta 0 while parked — freezes current value) |
| `vane_stuck` | vane | ✅ (delta 0 while parked — freezes current value) |
| `gyro_drift` | gyro | ✅ delta grows +0.01 rad/s² |
| `mag_drift` | mag | ✅ delta grows +0.03 rad/s |
| `baro_bias` | baro | ✅ constant +100 m offset |
| `gps_noise` | gps | ✅ Gaussian jitter σ=4 m/s |
| `accel_noise` | accel | ✅ Gaussian jitter σ=1.5 m/s² |

All faults: exactly one channel corrupted, others clean. `changed_channels()` proves it.
Freeze faults show 0 delta only because the c172p is parked (airspeed ≈ 5 kt, AoA ≈ constant). In flight they will diverge from truth.

---

### Phase 3 — The model (`model.py`) ✅ ARCHITECTURE DONE — exit criteria BLOCKED on training

- [x] Define `DetectorOutput` dataclass: `probs: dict[str,float]`, `top_sensor: str`,
      `confidence: float`, `reason: str` (+ `warmup: bool`)
- [x] `FaultDetector(nn.Module)`:
  - input window `[B, T, C]` (T = 50 steps, C = 7)
  - cross-channel consistency residuals
  - temporal consistency (rate-of-change, integration drift)
  - aggregation → per-sensor probability (softmax over sensors + `"none"`)
- [x] `reason` string generated from *which residual fired*
- [x] Handle warm-up (window not yet full) → `none` / confidence 0
- [x] Handle "no fault" → `top_sensor = none`

**Weights: RANDOM (user's explicit choice — "Literal random weights").**
`head` is `Linear(24→64) → ReLU → Linear(64→8)`, PyTorch default init, never
trained. The plan's rejected-alternatives table predicted the consequence
("Autoencoder with random weights … Not a model."). We ship the architecture;
training comes in ground-truth plan phase 2.

**Architecture — two deliberately separated parts**

| Part | Learnable? | Job |
|---|---|---|
| `ResidualEncoder` | **no** (fixed buffers) | temporal `std`/`\|slope\|`/`max\|step\|` per channel + 3 cross-channel rate disagreements → 24 features, plus a per-sensor anomaly `score` |
| `FaultDetector.head` | **yes, random** | 24 → 64 → 8 logits, softmax over 7 sensors + `none` |

The **abstention gate** (score < `threshold` → `none`) and **warm-up** are
deterministic and do NOT depend on the weights. They are what stop the demo
raising a random alarm on clean data. Gate is bypassed in `train()` mode so
`none` still receives a gradient later.

**Verified against live FlightGear** (`backend/phase3_test.py`):

| Condition | Encoder pick (deterministic) | Score | Gate | Head pick (random) | Exit criteria |
|---|---|---|---|---|---|
| clean, 60 windows | — | max 0.112 | CLOSED | — | ✅ `none` 60/60, no false alarm |
| `gps_noise` | **gps** ✅ | 18.199 | OPEN | accel ❌ | ❌ FAIL |
| `accel_noise` | **accel** ✅ | 8.861 | OPEN | baro ❌ | ❌ FAIL |
| `mag_drift` | **mag** ✅ | 0.756 | OPEN | gps ❌ | ❌ FAIL |
| `gyro_drift` | **gyro** ✅ | 0.252 | **CLOSED** | — | ❌ FAIL (gate too high) |
| `baro_bias` | baro (tied 0.000) | 0.000 | CLOSED | — | ❌ FAIL (see below) |
| `pitot_blockage` / `vane_stuck` | — | — | — | — | ⏸ not testable — aircraft parked |

**Exit criteria: BLOCKED.** The deterministic encoder names the right sensor in
3/3 testable faults; the random head then overrules it with a wrong guess.
Confidence on the correct sensor = 0.00 (need > 0.9). Training is the fix.

**Three blockers — status after calibration (FG restart NOT needed):**

1. **Random head** → wrong sensor whenever the gate opens.
   *Fix: training.* (User's deliberate choice for now.) **← the only one left**
2. ~~Gate threshold 0.5 above `gyro_drift`'s 0.252~~ → **FIXED**: threshold now
   **0.2**. Calibrated against live FG — clean peaks at 0.112 (still 27/27 and
   60/60 abstained, no false alarms), `gyro_drift` reaches 0.382 → gate OPEN,
   encoder picks `gyro`. `mag_drift` 0.756, `gps_noise` 18.2, `accel_noise` 8.9.
3. ~~`baro_bias` structurally undetectable~~ → **RESOLVED, no restart needed.**
   A constant offset changes no derivative, so a *level* comparison against any
   second altitude source fails: `alt`=146.06 ft vs `altind`=738.88 ft differ by
   a fixed 592.82 ft QNH offset, and the baseline would have to be learned.
   **Instead the fault is caught at onset** by the existing per-channel
   `max|step|` feature: activating `baro_bias` mid-stream gives encoder score
   **33.33 on `baro`** (only nonzero), gate OPEN. Detection is therefore
   *transient* — it alarms on the frame the button is pressed, then settles to
   `none` once the offset becomes the new normal. Persisting the alarm through
   a learned baseline is a training-time job.

**Files:** `backend/model.py` (encoder, `FaultDetector`, `DetectorOutput`,
`WindowBuffer`), `backend/phase3_test.py` (clean-score survey + per-fault run).

---

### Phase 4 — Server + HTML/JS dashboard  ✅ DONE (verified at protocol level)
- [x] `server.py`: static file server for `web/` + WebSocket endpoint `/ws`
      — **single port 8000**, static served through the handshake via
      `process_request`, so there is one URL to open and one port to firewall.
- [x] Message schema (JSON, one object per frame):
      `{ t, raw:{…}, channels:{…}, truth:{…}, attitude:{pitch,roll,hdg},
         verdict:{ top_sensor, confidence, reason, probs, sensor_scores,
                   gate_open, warmup }, active_fault, stats }`
      + `hello` on connect (channels, units, faults, threshold, weights) and
      `fault_state` / `status` / `error` control frames.
      `sensor_scores` + `gate_open` are **additive** extensions: without them
      the banner could only show the untrained head's answer with no
      explanation, which looks broken even though the encoder is correct.
- [x] `web/index.html` layout: attitude indicator · telemetry gauges ·
      live channel table · **model verdict banner** · fault-injection buttons
- [x] `web/app.js`: WebSocket client, reconnect w/ backoff (0.5 s → 5 s cap),
      ~30 Hz render throttled in a rAF loop, canvas attitude indicator,
      per-channel sparklines of the anomaly score
- [x] `web/style.css`: dark cockpit theme, readable at projector distance

**Exit criteria:** browser shows live telemetry + correct model verdict,
updating smoothly.

**Verified** — `phase4_ws_test.py` drives the browser's own protocol end to
end against the live server: **44 passed, 0 failed**.
- static: `/`, `/app.js`, `/style.css` all 200; `../../etc/passwd` → 404
- `hello` schema complete (7 channels, 7 presets, numeric threshold, units)
- telemetry at **≥10 Hz** (measured ~24 Hz), `frames_bad == 0`,
  `probs` covers 7 sensors + `none`, `sensor_scores` covers 7 sensors
- warm-up completes once the window fills
- `gyro_drift` → corrupts **exactly** `['gyro']`; `baro_bias` → exactly
  `['baro']`; `clear()` → `[]` both times
- unknown fault → `error` naming the bad preset; non-object JSON →
  `error: expected a JSON object`; server keeps streaming after both
- `node --check app.js` clean; every `$("id")` in app.js exists in index.html;
  JS `PRESET_CHANNEL` maps **exactly** match `faults.py` `PRESETS`

**⏳ still pending:** visual confirmation in an actual browser (layout, ADI
drawing, sparklines). The data/render contract is verified; the pixels are
not yet eyeballed.

**Bugs found and fixed during Phase 4**
1. `websockets` 17's `Request` has **no `extensions` attribute** (fields are
   `path, headers, method, protocol`) — passing `app` into the handler that
   way would have crashed on first connect. Now a closure over `app`.
2. `ResidualEncoder.forward` requires `[B,T,C]`; only `FaultDetector.forward`
   unsqueezes a bare `[T,C]`. Calling `model.encoder(t)` directly raised
   `expected [B,T,C], got shape (10, 7)` on every frame → `tensor.unsqueeze(0)`.
3. `age_seconds()` correctly returns `inf` before the first frame, but
   `"%.1fs" % inf` printed `NO SIGNAL for infs` → now formats as
   `never received any`.
4. A valid-JSON **non-object** frame was silently dropped; now answered with
   an error (the module docstring promised this).
5. `pkill -f "server.py --port"` **kills its own shell** — the command string
   contains the pattern. Kill by listener PID instead:
   `ss -tlnp | grep :8000 | grep -o 'pid=[0-9]*'`.

---

### Phase 5 — Integration + demo hardening  ⬅ NEXT
- [ ] `run.sh`: one command → starts FG with correct flags, backend, opens browser
- [ ] Failure handling: FG not running → clear message; port busy → clear message
- [ ] Fault buttons: `Pitot Blockage`, `Gyro Drift`, `Baro Bias`, `Clear`
- [ ] Rehearse: clean flight 20 s → inject fault → model names it → clear → recovers
- [ ] Latency measured: FG event → banner update (target < 150 ms)

**Exit:** a single command produces a demoable, repeatable result.

---

## 4. Explicit non-goals (this plan)

- No training of any model (C3)
- No Docker containers
- No .NET / SignalR / React — the WebSocket contract is designed so these can replace
  `server.py` + `web/` later without touching the AI
- No ANSYS / MATLAB
- No multi-user support
- No changes to anything under `docs/`

---

## 5. How this maps back to the ground-truth plan

| This plan | Ground-truth section |
|---|---|
| `model.py` detector | §4 AI Fault Detection Model |
| Sensor channels + fault injection | §4.2 Dataset Structure & fault labels |
| FG → Python → JSON | §3 Communication Protocols (stand-in for worker container) |
| `server.py` + `web/` | §6.4 SignalR Hub / §2.1 React Frontend (placeholder) |
| Swappable `nn.Module` interface | Phase 2 "Train model, export ONNX" (future) |

---

## 6. Open questions

1. Which aircraft to use for the demo? → **RESOLVED: `c172p`** (present in fgdata,
   real piston aircraft with working instruments; `ufo`/`aisim`/`Generic` are the others).
2. Does FG's generic protocol expose body rates and heading separately in 2024.1.7?
   → **RESOLVED in Phase 1: yes.** Body rates at `/orientation/{p,q,r}-body` (rad/s),
   heading at `/orientation/heading-deg`, AoA at `/orientation/alpha-deg`,
   sideslip at `/orientation/beta-deg` — all confirmed live. **No separate magnetic
   heading property is used**: `/orientation/heading-deg` is what we take as the
   "mag heading" channel for the demo (see open question 5).
3. Auto-start with engines running and airborne, or taxi/ground roll?
4. Should the fault also be *visible in the aircraft's behavior* (C1: FG owns physics),
   or is the fault purely in the instrument reading? → **default: instrument-only,
   which respects C1 and makes the detector's job meaningful.**
