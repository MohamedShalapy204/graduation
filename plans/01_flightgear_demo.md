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

### Phase 2 — Sensor channels + fault injection  ⬅ NEXT
- [ ] `sensor_channels.py`: truth state → channel dict (with units normalised)
- [ ] `faults.py`: injectors, **exactly one active at a time**
  - `freeze(channel, at_value)` — pitot blockage
  - `ramp(channel, rate)` — gyro / mag drift
  - `bias(channel, delta)` — baro error
  - `noise(channel, sigma)` — sensor degradation
- [ ] `FaultController` with `set_active(fault_type)` / `clear()`
- [ ] Log: clean channels vs corrupted channel, side by side

**Exit criteria:** toggling a fault in code visibly corrupts exactly one channel and no other.

---

### Phase 3 — The model (`model.py`)
- [ ] Define `DetectorOutput` dataclass: `probs: dict[str,float]`, `top_sensor: str`,
      `confidence: float`, `reason: str`
- [ ] `FaultDetector(nn.Module)`:
  - input window `[B, T, C]` (T ≈ 50 steps, C = channels)
  - cross-channel consistency residuals
  - temporal consistency (rate-of-change, integration drift)
  - aggregation → per-sensor probability (softmax over sensors + "none")
- [ ] `reason` string is generated from *which residual fired*, e.g.
      `"pitot frozen: airspeed variance≈0 while GPS groundspeed=42 m/s"`
- [ ] Handle warm-up (window not yet full) → emit `none` / low confidence
- [ ] Handle "no fault" → all probs low, top_sensor = `none`

**Exit criteria:** with each fault type injected in turn, the model names the correct
sensor with confidence > 0.9, and returns `none` on clean data with no false alarm
over ≥60 s of flight.

---

### Phase 4 — Server + HTML/JS dashboard
- [ ] `server.py`: static file server for `web/` + WebSocket endpoint `/ws`
- [ ] Message schema (JSON, one object per frame):
      `{ t, channels:{...}, truth:{...}, verdict:{ top_sensor, confidence, reason, probs },
         active_fault }`
- [ ] `web/index.html` layout: attitude indicator · telemetry gauges ·
      live channel table · **model verdict banner** · fault-injection buttons
- [ ] `web/app.js`: WebSocket client, reconnect w/ backoff, ~30 Hz render,
      canvas attitude indicator, sparklines for residuals
- [ ] `web/style.css`: dark cockpit theme, readable at projector distance

**Exit criteria:** browser shows live telemetry + correct model verdict, updating smoothly.

---

### Phase 5 — Integration + demo hardening
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
