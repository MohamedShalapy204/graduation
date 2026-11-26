"""Fault injection: corrupt EXACTLY ONE channel, instrument-only.

Phase 2 of plans/01_flightgear_demo.md.

Ground rules (confirmed with the user)
--------------------------------------
* FlightGear owns the physics. A fault here NEVER touches the aircraft --
  it only corrupts the *reading* handed to the detector. The sim keeps flying
  straight while the sensor lies.
* Single-fault assumption: ``FaultController`` holds at most one active fault.
  Asking for a second one silently replaces the first.
* ``apply()`` copies the channel dict and writes exactly one key, so a fault
  can never leak into a neighbouring channel (that is the Phase 2 exit
  criterion, and ``changed_channels()`` exists to prove it).

Injector types (plan section 3)
-------------------------------
    freeze(channel, at_value)  pitot blockage -- reading sticks
    ramp(channel, rate)        gyro / mag drift -- offset grows with time
    bias(channel, delta)       baro error -- constant offset
    noise(channel, sigma)      sensor degradation -- gaussian jitter

Named recipes (``PRESETS``) are what the demo buttons call; each maps to one
injector on one channel.

Running this file directly
--------------------------
    python backend/faults.py --fault pitot_blockage --seconds 10

prints clean vs corrupted channels side by side and then asserts that exactly
one channel moved. Needs FlightGear running (see plans/01_flightgear_demo.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from sensor_channels import CHANNELS, CHANNEL_META, build_channels

FAULT_TYPES: Tuple[str, ...] = ("freeze", "ramp", "bias", "noise")


# ---------------------------------------------------------------------------
# injectors
# ---------------------------------------------------------------------------
class FreezeInjector:
    """Reading sticks at one value (blocked pitot, stuck vane).

    With ``at_value=None`` the value present on the FIRST frame after
    activation is latched -- physically, the moment the blockage happened.
    """

    kind = "freeze"

    def __init__(self, at_value: Optional[float] = None) -> None:
        self.at_value = at_value
        self._latched: Optional[float] = None

    def apply(self, value: float, t: float) -> float:
        if self._latched is None:
            self._latched = value if self.at_value is None else float(self.at_value)
        return self._latched

    def describe(self) -> str:
        if self._latched is None:
            return "frozen (awaiting first frame)"
        return f"frozen at {self._latched:.4f}"


class RampInjector:
    """Offset grows linearly with sim time (gyro / magnetic drift)."""

    kind = "ramp"

    def __init__(self, rate: float) -> None:
        self.rate = float(rate)
        self._t0: Optional[float] = None
        self._offset: float = 0.0

    def apply(self, value: float, t: float) -> float:
        if self._t0 is None:
            self._t0 = t
        self._offset = self.rate * (t - self._t0)
        return value + self._offset

    def describe(self) -> str:
        if self._t0 is None:
            return f"drifting at {self.rate:+.5f}/s (not started)"
        return f"drifting at {self.rate:+.5f}/s (offset {self._offset:+.4f})"


class BiasInjector:
    """Constant offset (altimeter setting error)."""

    kind = "bias"

    def __init__(self, delta: float) -> None:
        self.delta = float(delta)

    def apply(self, value: float, t: float) -> float:
        return value + self.delta

    def describe(self) -> str:
        return f"biased {self.delta:+.4f}"


class NoiseInjector:
    """Gaussian jitter (degraded sensor, spike-prone GPS)."""

    kind = "noise"

    def __init__(self, sigma: float, rng: Optional[random.Random] = None) -> None:
        self.sigma = float(sigma)
        self._rng = rng or random.Random()

    def apply(self, value: float, t: float) -> float:
        return value + self._rng.gauss(0.0, self.sigma)

    def describe(self) -> str:
        return f"noisy sigma={self.sigma:.4f}"


# ---------------------------------------------------------------------------
# spec + presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FaultSpec:
    """What is active: which injector, on which channel, with what params."""

    fault_type: str
    channel: str
    params: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        inner = f"{self.channel}, {args}" if args else self.channel
        return f"{self.fault_type}({inner})"


# Named recipes. Rates/deltas chosen to be visible on screen within a few
# seconds without being absurd: gyro 0.01 rad/s^2 reaches 0.1 rad/s after 10 s,
# mag 0.03 rad/s is ~1.7 deg/s, baro +100 m is unmissable.
PRESETS: Dict[str, Dict[str, object]] = {
    "pitot_blockage": {"fault_type": "freeze", "channel": "pitot"},
    "vane_stuck": {"fault_type": "freeze", "channel": "vane"},
    "gyro_drift": {"fault_type": "ramp", "channel": "gyro", "rate": 0.01},
    "mag_drift": {"fault_type": "ramp", "channel": "mag", "rate": 0.03},
    "baro_bias": {"fault_type": "bias", "channel": "baro", "delta": 100.0},
    "gps_noise": {"fault_type": "noise", "channel": "gps", "sigma": 4.0},
    "accel_noise": {"fault_type": "noise", "channel": "accel", "sigma": 1.5},
}


def _make_injector(spec: FaultSpec, rng: random.Random):
    if spec.fault_type == "freeze":
        return FreezeInjector(at_value=spec.params.get("at_value"))
    if spec.fault_type == "ramp":
        return RampInjector(rate=spec.params["rate"])
    if spec.fault_type == "bias":
        return BiasInjector(delta=spec.params["delta"])
    if spec.fault_type == "noise":
        return NoiseInjector(sigma=spec.params["sigma"], rng=rng)
    raise ValueError(f"unknown fault type {spec.fault_type!r} (known: {FAULT_TYPES})")


# ---------------------------------------------------------------------------
# controller
# ---------------------------------------------------------------------------
class FaultController:
    """Holds at most one fault and applies it to exactly one channel."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()
        self._spec: Optional[FaultSpec] = None
        self._injector = None

    # -- state -----------------------------------------------------------
    @property
    def active(self) -> Optional[FaultSpec]:
        return self._spec

    @property
    def active_channel(self) -> Optional[str]:
        return self._spec.channel if self._spec else None

    @property
    def is_active(self) -> bool:
        return self._spec is not None

    # -- control ---------------------------------------------------------
    def set_active(
        self,
        fault_type: str,
        channel: Optional[str] = None,
        **params: float,
    ) -> FaultSpec:
        """Activate one fault. Replaces any fault already active.

        Two ways to call it:

            set_active("pitot_blockage")              # preset name, channel implied
            set_active("freeze", channel="pitot")     # explicit injector + channel

        """
        if fault_type in PRESETS:
            preset = dict(PRESETS[fault_type])
            preset.update(params)
            if channel is not None:
                preset["channel"] = channel
            fault_type = str(preset.pop("fault_type"))
            channel = str(preset.pop("channel"))
            params = {k: float(v) for k, v in preset.items()}
        elif channel is None:
            raise ValueError(
                f"fault_type {fault_type!r} is not a preset; pass channel=... "
                f"(presets: {sorted(PRESETS)})"
            )

        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r} (known: {list(CHANNELS)})")
        if fault_type not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {fault_type!r} (known: {list(FAULT_TYPES)})")

        spec = FaultSpec(fault_type=fault_type, channel=channel, params=params)
        self._spec = spec
        self._injector = _make_injector(spec, self._rng)
        return spec

    def clear(self) -> None:
        """Back to clean data."""
        self._spec = None
        self._injector = None

    # -- application -----------------------------------------------------
    def apply(self, channels: Mapping[str, float], t: float) -> Dict[str, float]:
        """Return a copy of *channels* with the active fault applied.

        No fault -> an equal copy. With a fault -> exactly one key rewritten.
        """
        out = dict(channels)
        if self._spec is None:
            return out
        ch = self._spec.channel
        if ch not in out:
            raise KeyError(
                f"fault channel {ch!r} missing from channels {sorted(out)}"
            )
        out[ch] = self._injector.apply(out[ch], t)
        return out

    def describe(self) -> str:
        if self._spec is None:
            return "no fault (clean)"
        return f"{self._spec}  [{self._injector.describe()}]"


# ---------------------------------------------------------------------------
# reporting helpers (the plan asks for a clean-vs-corrupted side-by-side log)
# ---------------------------------------------------------------------------
def changed_channels(
    clean: Mapping[str, float], corrupted: Mapping[str, float]
) -> List[str]:
    """Channels whose value actually differs -- the Phase 2 proof."""
    return [c for c in CHANNELS if clean.get(c) != corrupted.get(c)]


def format_side_by_side(
    clean: Mapping[str, float],
    corrupted: Mapping[str, float],
    spec: Optional[FaultSpec] = None,
    note: str = "",
) -> str:
    """Aligned ``clean | corrupted`` table with the faulty channel marked."""
    width = max(len(c) for c in CHANNELS)
    lines = []
    header = f"  {'channel':<{width}}  {'clean':>12}  {'corrupted':>12}  note"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for c in CHANNELS:
        unit = CHANNEL_META[c]["unit"]
        clean_v = clean[c]
        dirty_v = corrupted[c]
        differs = clean_v != dirty_v
        if spec is not None and c == spec.channel:
            note = spec.fault_type
            if spec.fault_type == "freeze":
                note = f"freeze @ {dirty_v:.4f}"
            elif spec.fault_type == "ramp":
                note = f"drift {dirty_v - clean_v:+.4f} {unit}"
            elif spec.fault_type == "bias":
                note = f"bias {dirty_v - clean_v:+.4f} {unit}"
            elif spec.fault_type == "noise":
                note = f"noise {dirty_v - clean_v:+.4f} {unit}"
            marker = "  <-- FAULT"
        else:
            marker = "" if not differs else "  <-- UNEXPECTED"
            note = "unchanged" if not differs else "changed!"
        lines.append(
            f"  {c:<{width}}  {clean_v:12.4f}  {dirty_v:12.4f}  {note}{marker}"
        )
    if note:
        lines.append(f"  ({note})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# demo / self-test
# ---------------------------------------------------------------------------
def _demo(fault: str, seconds: float, port: int) -> int:
    import time

    from fg_reader import FGReader

    controller = FaultController()
    try:
        spec = controller.set_active(fault)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"available faults: {sorted(PRESETS)}")
        return 1

    reader = FGReader(port=port)
    try:
        reader.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"fault: {controller.describe()}")
    print(f"listening on udp://localhost:{port} for {seconds:.0f}s ...")
    print("(corrupts the READING only -- FlightGear keeps flying normally)\n")

    deadline = time.monotonic() + seconds
    last_print = 0.0
    last_clean = last_dirty = None
    worst = None

    while time.monotonic() < deadline:
        state = reader.poll(timeout=0.25)
        if state is None:
            continue
        t = state["t"]
        clean = build_channels(state)
        dirty = controller.apply(clean, t)
        last_clean, last_dirty = clean, dirty

        moved = changed_channels(clean, dirty)
        if moved and worst is None:
            worst = moved
        elif moved and worst != moved:
            # a fault must never wander onto a second channel
            print(f"FAIL: fault moved onto extra channels: {moved} (was {worst})")
            controller.clear()
            reader.close()
            return 1

        now = time.monotonic()
        if now - last_print >= 2.0:
            last_print = now
            print(f"--- t = {t:.3f} s ---")
            print(format_side_by_side(clean, dirty, spec))
            print()

    ok, bad, _ = reader.stats
    reader.close()

    if ok == 0:
        print("FAIL: no frames received. Is FlightGear running with "
              "--generic=socket,out,...,udp,telemetry ?")
        return 1

    print(f"frames ok={ok} bad={bad}")
    if last_clean is None:
        print("FAIL: no frames parsed")
        return 1

    moved = changed_channels(last_clean, last_dirty)
    print(f"channels changed by the fault: {moved}")
    if len(moved) != 1:
        print(f"FAIL: expected exactly 1 corrupted channel, got {len(moved)}")
        return 1
    if moved[0] != spec.channel:
        print(f"FAIL: corrupted {moved[0]!r}, expected {spec.channel!r}")
        return 1
    print(f"OK: exactly one channel corrupted ({moved[0]})")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fault", default="pitot_blockage", choices=sorted(PRESETS))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--port", type=int, default=5510)
    args = ap.parse_args()
    raise SystemExit(_demo(args.fault, args.seconds, args.port))
