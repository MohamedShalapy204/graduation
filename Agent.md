# Agent.md — Project Operating Rules

This file tells any AI agent (or human) working in this repository **how to work here**.
It is short on purpose. Read it before touching anything.

---

## 1. Source of Truth

**The ONLY ground-truth document is:**

```
docs/new/system_architecture_implementation_plan(1).md
```

It defines the target system: React frontend, .NET 8/9 backend + orchestrator,
Docker per-user worker containers, AI fault detector, PostgreSQL/EF Core schema,
SignalR telemetry hub, and the 12-week roadmap.

### Rules about other files

- Everything under `docs/old/` is **legacy/junk**. Do **not** read it.
- Other PDFs in `docs/new/` are **reference only**. Do not treat them as requirements.
- **Do not delete or rewrite any file in `docs/`.** It belongs to the user.

If a task appears to contradict the ground-truth plan, **ask before proceeding.**

---

## 2. Team Roles (decides who owns what)

| Person | Responsibility |
|---|---|
| **Mohamed** | AI models + Backend |
| (other members) | Frontend, MATLAB/ANSYS dynamics — *not managed by this agent* |

Consequence: this agent works primarily in **Python (AI + backend)**.
Anything React/.NET/MATLAB is **out of scope unless explicitly asked**.

---

## 3. Current Demo Track (temporary, time-critical)

A doctor demo is in progress. This **does not replace** the ground-truth plan — it is
a fast vertical slice built *on top of* FlightGear.

- **FlightGear owns physics, simulation, and visualization.** We never modify FG.
- **Python owns the AI model and the backend** for this demo.
- **No model training right now.** Demo only.
- **Model reads sensors directly from FlightGear** and predicts *which* sensor is faulty.
- **One fault at a time** (single-fault assumption).
- **Output is displayed in HTML/JS.**

Full detail: `plans/01_flightgear_demo.md`

---

## 4. Environment

| Item | Value |
|---|---|
| OS | Linux Mint |
| FlightGear | `flightgear-2024.1.7-linux-amd64.AppImage` (repo root) |
| FG data root | `~/.fgfs/fgdata_2024_1` (note: folder is `Protocol/`, capital P) |
| Python | 3.12.3 |
| Venv | `.venv/` at repo root — **copy of** `~/projects/rl-project/.venv` |
| torch | 2.15.0+cu132 — **CUDA unavailable, run on CPU** |
| GPU | none → never assume `cuda` |

**Never** install into or modify `~/projects/rl-project/.venv`. It belongs to another project.

### Venv usage

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 5. Hard Rules

1. **Do not delete, move, or overwrite user files.** Ask first. (`squashfs-root/` was
   agent-generated build output — agent-generated artifacts may be cleaned by the agent,
   but only those it created itself.)
2. **Do not train models** unless explicitly told to.
3. **Do not install packages system-wide.** Always inside `.venv`.
4. **Do not touch `docs/`.**
5. **Keep the pipeline swappable.** Whatever runs in the demo must have the same
   interface as the eventual trained model, so Phase 2 of the ground-truth plan
   (AI Model Training) drops in without rework.
6. **Prefer one working end-to-end path over many half-finished parts.**
7. **State assumptions explicitly** rather than silently guessing.

---

## 6. Plans Folder

`plans/` holds ordered, checked-off implementation plans.

- `plans/01_flightgear_demo.md` — current active plan
- Add `02_*.md`, `03_*.md`, … as new phases start.

Mark tasks `[x]` only when **verified working**, not merely written.

---

## 7. Definition of Done (any task)

- It runs end-to-end with a command you can show someone.
- Failure modes are handled (process dies → clear message, not a silent hang).
- The plan file is updated to reflect reality.
- No unrelated files were touched.
