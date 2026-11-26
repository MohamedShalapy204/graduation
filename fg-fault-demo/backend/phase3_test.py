"""Phase 3 verification against live FlightGear.

Two questions:
  1. What does a CLEAN signal score? (sets the abstention threshold honestly)
  2. With each fault injected, what does the model say? (exit criteria)

Run:  python phase3_test.py [--seconds 20]
Needs FlightGear running with the telemetry socket.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from fg_reader import FGReader
from sensor_channels import CHANNELS, build_channels
from faults import FaultController, PRESETS
from model import (
    DEFAULT_WINDOW,
    MIN_WARMUP,
    N_CLASSES,
    SENSORS,
    FaultDetector,
    WindowBuffer,
)


def collect(seconds: float, port: int):
    """Pull clean channel frames from live FG for *seconds*."""
    reader = FGReader(port=port)
    try:
        reader.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return None
    frames = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = reader.poll(timeout=0.25)
        if state is None:
            continue
        frames.append((state["t"], build_channels(state)))
    ok, bad, _ = reader.stats
    reader.close()
    print(f"  collected {len(frames)} frames (ok={ok} bad={bad}) in {seconds:.0f}s")
    return frames


def window_from(frames, start: int, size: int):
    buf = WindowBuffer(size=size)
    for _, ch in frames[start : start + size]:
        buf.push(ch)
    return buf.tensor()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--port", type=int, default=5510)
    ap.add_argument("--fault", default=None, choices=sorted(PRESETS))
    args = ap.parse_args()

    torch.manual_seed(0)
    model = FaultDetector()
    model.eval()
    print(f"weights: {model.weight_status()}")
    print(f"params : {sum(p.numel() for p in model.parameters())}\n")

    print(f"collecting {args.seconds:.0f}s of CLEAN data from udp://localhost:{args.port} ...")
    frames = collect(args.seconds, args.port)
    if not frames or len(frames) < DEFAULT_WINDOW + 5:
        print("FAIL: not enough frames")
        return 1

    # ---- 1. clean score distribution (sets threshold) --------------------
    print("\n== 1. CLEAN data: per-sensor residual scores ==")
    enc = model.encoder
    all_scores = []
    for start in range(0, len(frames) - DEFAULT_WINDOW, 10):
        w = window_from(frames, start, DEFAULT_WINDOW)
        if w is None or w.shape[0] < MIN_WARMUP:
            continue
        with torch.no_grad():
            _, score, *_ = enc(w.unsqueeze(0))
        all_scores.append(score.squeeze(0))
    stacked = torch.stack(all_scores)  # [N, 7]
    print(f"  {stacked.shape[0]} windows of {DEFAULT_WINDOW} frames")
    print(f"  {'sensor':<8} {'mean':>8} {'p95':>8} {'max':>8}")
    clean_max_per_window = stacked.max(dim=1).values
    for i, name in enumerate(SENSORS):
        col = stacked[:, i]
        p95 = torch.quantile(col, 0.95)
        print(f"  {name:<8} {col.mean():8.3f} {p95:8.3f} {col.max():8.3f}")
    print(f"  worst window score: {clean_max_per_window.max():.3f} "
          f"(mean {clean_max_per_window.mean():.3f})")

    # threshold must sit above clean data or the demo false-alarms
    observed = float(clean_max_per_window.max())
    suggested = max(0.5, observed * 1.5)
    print(f"\n  current threshold = {model.threshold}")
    print(f"  observed clean max = {observed:.3f}")
    print(f"  suggested threshold = {suggested:.3f} (clean max x1.5)")
    if observed >= model.threshold:
        print(f"  !! threshold {model.threshold} is BELOW clean max -> false alarms")

    # ---- 2. clean windows through the full model ------------------------
    print("\n== 2. Full model on CLEAN windows (must abstain) ==")
    abstained = 0
    total = 0
    for start in range(0, len(frames) - DEFAULT_WINDOW, 10):
        w = window_from(frames, start, DEFAULT_WINDOW)
        if w is None:
            continue
        out = model(w)
        total += 1
        if out.top_sensor == "none":
            abstained += 1
    print(f"  abstained on {abstained}/{total} clean windows "
          f"({100.0 * abstained / max(total, 1):.0f}%)")

    # ---- 3. each fault, injected over the tail of the stream -------------
    if args.fault is None:
        print("\n(no --fault given; run again with e.g. --fault baro_bias)")
        print("\nOK (clean-side verified)")
        return 0

    print(f"\n== 3. Fault: {args.fault} ==")
    spec_params = PRESETS[args.fault]
    ctl = FaultController()
    spec = ctl.set_active(args.fault)
    print(f"  active: {ctl.describe()}")

    # re-collect WITH the fault active
    print(f"  collecting {args.seconds:.0f}s with fault active ...")
    faulty = collect(args.seconds, args.port)
    if not faulty or len(faulty) < DEFAULT_WINDOW + 5:
        print("FAIL: not enough frames with fault")
        return 1

    verdicts = []
    scores_seen = []
    for start in range(0, len(faulty) - DEFAULT_WINDOW, 10):
        clean_buf = WindowBuffer()
        dirty_buf = WindowBuffer()
        for t, ch in faulty[start : start + DEFAULT_WINDOW]:
            clean_buf.push(ch)  # truth, unfaulted
            dirty_buf.push(ctl.apply(ch, t))  # what the model sees
        w = dirty_buf.tensor()
        if w is None or w.shape[0] < MIN_WARMUP:
            continue
        with torch.no_grad():
            _, score, *_ = model.encoder(w.unsqueeze(0))
        scores_seen.append(score.squeeze(0))
        out = model(w)
        verdicts.append(out)

    if not verdicts:
        print("FAIL: no windows")
        return 1

    # residual scores WITH the fault active: this is the deterministic encoder,
    # independent of the random head. If these spike on the right sensor the
    # encoder is fine and the head is the only thing blocking exit criteria.
    st = torch.stack(scores_seen)
    expected = spec.channel
    print("\n  residual scores WITH fault active (deterministic encoder):")
    print(f"  {'sensor':<8} {'mean':>8} {'max':>8}")
    for i, name in enumerate(SENSORS):
        col = st[:, i]
        mark = "  <-- expected" if name == expected else ""
        print(f"  {name:<8} {col.mean():8.3f} {col.max():8.3f}{mark}")
    enc_best = SENSORS[int(st.mean(dim=0).argmax())]
    print(f"  encoder's own pick (highest mean residual): {enc_best}")
    print(f"  gate: max score {float(st.max()):.3f} vs threshold {model.threshold}"
          f" -> {'OPEN' if float(st.max()) >= model.threshold else 'CLOSED'}")

    from collections import Counter

    counts = Counter(v.top_sensor for v in verdicts)
    print(f"  windows scored: {len(verdicts)}")
    for sensor, n in counts.most_common():
        print(f"    {sensor:<8} {n:4d}  ({100.0 * n / len(verdicts):.0f}%)")
    print(f"  example reason: {verdicts[-1].reason}")

    hit = counts.get(expected, 0)
    rate = hit / len(verdicts)
    print(f"\n  expected sensor : {expected}")
    print(f"  detected as such: {hit}/{len(verdicts)} ({100.0 * rate:.0f}%)")
    print(f"  exit criteria (correct sensor, conf > 0.9): "
          f"{'PASS' if rate > 0.5 else 'FAIL'}")

    conf_ok = max(
        (v.confidence for v in verdicts if v.top_sensor == expected), default=0.0
    )
    print(f"  best confidence on correct sensor: {conf_ok:.2f} "
          f"(need > 0.90) -> {'PASS' if conf_ok > 0.9 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
