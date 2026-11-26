"""Truth state (one FlightGear frame) -> normalised sensor channel dict.

Phase 2 of plans/01_flightgear_demo.md.

The detector in Phase 3 sees *channels*, not raw FG properties. This module is
the single place where a raw frame becomes that vector, so unit conversion
happens exactly once, here, and nowhere else.

Channels (7, matching the plan's sensor table)

    id      value                         unit     FG source
    ------  ----------------------------  -------  ------------------------------------
    pitot   indicated airspeed            m/s      ias (kt)
    gps     ground-speed magnitude        m/s      hypot(speed-north, speed-east) (ft/s)
    baro    altitude                      m        altitude-ft  (altimeter fallback)
    gyro    body-rate magnitude           rad/s    hypot(p-body, q-body, r-body)
    mag     heading                       rad      heading-deg, wrapped to [0, 2*pi)
    vane    angle of attack               rad      alpha-deg
    accel   body specific-force magnitude m/s^2    hypot(pilot x,y,z-accel) (ft/s^2)

Every channel has at least one independent physical witness -- that is WHY these
sensors were chosen (plan section 2):

    pitot <-> gps          both measure how fast the aircraft is moving
    baro  <-> (gps alt, vertical speed)
    gyro  <-> finite difference of the Euler angles
    mag   <-> integrated yaw rate
    vane  <-> atan2(w, u) from body velocity
    accel <-> finite difference of velocity

Limitation worth stating out loud: the detector's tensor is built from THESE
seven scalars only, so a residual that needs pitch/roll or a body-velocity
vector cannot be computed from the tensor alone. If Phase 3 needs those, add
properties to protocol/telemetry.xml and re-verify against live FG (ask the
user first -- that means touching FG's launch command).

Units are SI so Phase 3 thresholds live in one system.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

# -- unit conversions -------------------------------------------------------
KT_TO_MS = 0.514444  # knot  -> m/s
FT_TO_M = 0.3048  # foot  -> m
FT_S_TO_M_S = 0.3048  # ft/s  -> m/s  (same factor, named for clarity)
DEG_TO_RAD = math.pi / 180.0

# FG reports this for altitude during the first ~2 s of a session (verified in
# Phase 1). It is not a real altitude -- below-sea-level terrain never gets
# near -9999 ft -- so fall back to the altimeter's indicated altitude.
STARTUP_SENTINEL = -9999.0

# Order matters: this is the channel axis order of the model's [B, T, C] tensor.
CHANNELS: Tuple[str, ...] = ("pitot", "gps", "baro", "gyro", "mag", "vane", "accel")

# Keys a frame must carry. Loud failure beats a silent 0.0 that later looks
# like a sensor fault -- if someone edits protocol/telemetry.xml, this fires.
REQUIRED_KEYS: Tuple[str, ...] = (
    "ias",
    "vng",
    "veg",
    "alt",
    "altind",
    "p",
    "q",
    "r",
    "hdg",
    "alpha",
    "ax",
    "ay",
    "az",
)

# Documentation + logging only. `witness` is the redundancy partner from the
# plan's sensor table.
CHANNEL_META: Dict[str, Dict[str, str]] = {
    "pitot": {"unit": "m/s", "fg": "ias (kt)", "witness": "gps"},
    "gps": {"unit": "m/s", "fg": "speed-north/east (ft/s)", "witness": "pitot"},
    "baro": {"unit": "m", "fg": "altitude-ft", "witness": "gps altitude + vs"},
    "gyro": {"unit": "rad/s", "fg": "p,q,r-body", "witness": "d/dt Euler angles"},
    "mag": {"unit": "rad", "fg": "heading-deg", "witness": "integrated yaw rate"},
    "vane": {"unit": "rad", "fg": "alpha-deg", "witness": "atan2(w, u)"},
    "accel": {"unit": "m/s^2", "fg": "pilot x,y,z-accel-fps_sec", "witness": "d/dt velocity"},
}


def wrap2pi(angle_rad: float) -> float:
    """Wrap an angle into [0, 2*pi).

    Python's % with a positive divisor already returns a non-negative result,
    so no extra branch is needed.
    """
    return angle_rad % (2.0 * math.pi)


def build_channels(state: Mapping[str, float]) -> Dict[str, float]:
    """Convert one raw FG frame into the normalised channel dict.

    Returns a fresh dict keyed by exactly ``CHANNELS``. Pure: the input frame
    is never mutated (fault injection must not be able to reach truth).

    Raises ValueError if a required key is missing -- a protocol mismatch must
    be loud, not silently read as 0.0.
    """
    missing = [key for key in REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"frame missing required keys: {missing}")

    # pitot: indicated airspeed, kt -> m/s
    pitot = state["ias"] * KT_TO_MS

    # gps: ground-speed magnitude from the north/east components, ft/s -> m/s
    gps = math.hypot(state["vng"], state["veg"]) * FT_S_TO_M_S

    # baro: altitude, ft -> m. Skip FG's startup sentinel.
    alt_ft = state["alt"]
    if alt_ft <= STARTUP_SENTINEL + 1.0:
        alt_ft = state["altind"]
    baro = alt_ft * FT_TO_M

    # gyro: body-rate magnitude, rad/s (FG already reports rad/s)
    gyro = math.sqrt(state["p"] ** 2 + state["q"] ** 2 + state["r"] ** 2)

    # mag: heading, deg -> rad, wrapped (a 359->1 deg turn is a small step)
    mag = wrap2pi(state["hdg"] * DEG_TO_RAD)

    # vane: angle of attack, deg -> rad
    vane = state["alpha"] * DEG_TO_RAD

    # accel: body specific-force magnitude, ft/s^2 -> m/s^2 (reads ~1 g at rest)
    accel = math.sqrt(state["ax"] ** 2 + state["ay"] ** 2 + state["az"] ** 2) * FT_S_TO_M_S

    return {
        "pitot": pitot,
        "gps": gps,
        "baro": baro,
        "gyro": gyro,
        "mag": mag,
        "vane": vane,
        "accel": accel,
    }


def units() -> Dict[str, str]:
    """``{channel: unit}`` -- handy for labels in the HTML dashboard."""
    return {name: meta["unit"] for name, meta in CHANNEL_META.items()}


def _demo(host: str = "localhost", port: int = 5510, seconds: float = 8.0) -> int:
    """Print live channels next to their raw FG sources so units can be eyeballed."""
    from fg_reader import FGReader  # local import: keeps this module importable standalone

    reader = FGReader(host, port)
    try:
        reader.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"reading udp://{host}:{port} for {seconds:.0f}s ...")
    print(f"{'t':>8} " + " ".join(f"{c:>9}" for c in CHANNELS) + "   (SI units)")
    print(f"{'(s)':>8} " + " ".join(f"{CHANNEL_META[c]['unit']:>9}" for c in CHANNELS))

    import time

    deadline = time.monotonic() + seconds
    shown = 0
    while time.monotonic() < deadline:
        state = reader.poll(timeout=0.25)
        if state is None:
            continue
        if shown % 6 == 0:
            ch = build_channels(state)
            print(
                f"{state['t']:8.3f} "
                + " ".join(f"{ch[c]:9.4f}" for c in CHANNELS)
                + f"   raw: ias={state['ias']:.2f}kt alt={state['alt']:.1f}ft "
                f"hdg={state['hdg']:.1f}deg alpha={state['alpha']:.2f}deg"
            )
        shown += 1

    ok, bad, last_bad = reader.stats
    reader.close()
    print(f"\nframes ok={ok} bad={bad}")
    if ok == 0:
        print("FAIL: no frames. Is FlightGear running with "
              "--generic=socket,out,...,udp,telemetry ?")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5510)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()
    raise SystemExit(_demo(args.host, args.port, args.seconds))
