"""Phase 3 -- the fault detector (plan section 3).

What this is
------------
A genuine ``torch.nn.Module`` that takes a window of sensor channels ``[B, T, C]``
(T = 50 frames, C = 7) and reports which sensor is faulty.

Two parts, deliberately separated:

1. ``ResidualEncoder`` -- FIXED, no learnable parameters. Computes the
   cross-channel and temporal consistency residuals that plan section 1 calls
   "analytical redundancy". This is deterministic maths: it is meaningful with
   or without training, and it never changes.

2. ``FaultDetector.head`` -- a small MLP, ``nn.Linear -> ReLU -> nn.Linear``,
   mapping residual features to logits over 8 classes (7 sensors + "none").

WEIGHTS ARE RANDOM.  (Confirmed choice: "Literal random weights".)
``head`` uses PyTorch's default initialisation and has never been trained.
Consequences, stated up front rather than discovered during the demo:

    * the residual encoder still works -- "is anything odd?" is answered by maths
    * the classifier head GUESSES -- "which sensor?" is meaningless right now
    * therefore Phase 3's exit criteria (correct sensor, confidence > 0.9) is
      BLOCKED until training happens (ground-truth plan phase 2)
    * the plan's rejected-alternatives table already predicted this:
      "Autoencoder with random weights -- reconstruction error would be
      meaningless. Not a model."  We are shipping the architecture, not a
      trained detector.

What DOES work today
--------------------
* warm-up: window not yet full -> ``top_sensor = "none"``, confidence 0
* abstention: no residual reaches the threshold -> ``top_sensor = "none"``
  (this is a deterministic gate, not the network -- it is what stops the demo
  from raising a random alarm on perfectly clean data)
* ``reason`` strings, generated from which residual actually fired

Training later replaces ``head`` only. The encoder and the public interface
(``DetectorOutput``) stay put, so Phase 2 of the ground-truth plan drops in
without rework -- rule 5 in Agent.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from sensor_channels import CHANNELS

# -- shape of the problem ---------------------------------------------------
SENSORS: Tuple[str, ...] = CHANNELS  # order = tensor channel axis
CLASSES: Tuple[str, ...] = CHANNELS + ("none",)
N_CHANNELS: int = len(SENSORS)  # 7
N_CLASSES: int = len(CLASSES)  # 8

DEFAULT_WINDOW: int = 50  # plan: T ~ 50 steps
MIN_WARMUP: int = 10  # below this we refuse to answer

# Per-channel engineering scales: a residual of 1.0 here means "one normal
# unit of this phenomenon for this sensor". These are feature scaling, not
# weights -- they are what makes the residual features comparable across a
# channel measured in m/s and one measured in rad/s.
# order = (scale for std, scale for |slope|, scale for max step)
_SCALES: Dict[str, Tuple[float, float, float]] = {
    "pitot": (0.60, 0.15, 0.80),  # m/s,   m/s^2,  m/s
    "gps": (0.60, 0.15, 0.80),  # m/s,   m/s^2,  m/s
    "baro": (2.00, 0.60, 3.00),  # m,     m/s,    m
    "gyro": (0.02, 0.05, 0.04),  # rad/s, rad/s^2,rad/s
    "mag": (0.02, 0.03, 0.04),  # rad,   rad/s,  rad
    "vane": (0.02, 0.03, 0.04),  # rad,   rad/s,  rad
    "accel": (0.40, 1.00, 0.80),  # m/s^2, m/s^3,  m/s^2
}

# Cross-channel pairs: each endpoint witnesses the other. A disagreement in
# their RATE of change means one of them is lying.
_CROSS: Tuple[Tuple[str, str], ...] = (
    ("pitot", "gps"),  # two independent measures of how fast we move
    ("mag", "gyro"),  # heading should change iff there is a body rate
    ("accel", "gyro"),  # you do not accelerate hard without rotating
)

# 3 temporal features per channel + one feature per cross pair
N_FEATURES: int = 3 * N_CHANNELS + len(_CROSS)  # 24


@dataclass(frozen=True)
class DetectorOutput:
    """What the model says about one window.

    Attributes:
        probs: probability per class, sums to 1.0 over ``CLASSES``
        top_sensor: the winning class (a sensor id, or ``"none"``)
        confidence: probability attached to ``top_sensor``
        reason: human-readable evidence, generated from which residual fired
        warmup: True while the window is still filling
    """

    probs: Dict[str, float]
    top_sensor: str
    confidence: float
    reason: str
    warmup: bool = False

    def __str__(self) -> str:
        tag = " (warming up)" if self.warmup else ""
        return f"{self.top_sensor} @ {self.confidence:.2f}{tag} -- {self.reason}"


# ---------------------------------------------------------------------------
# fixed feature extraction
# ---------------------------------------------------------------------------
class ResidualEncoder(nn.Module):
    """Deterministic residuals. No learnable parameters, by design.

    Per channel (temporal):  std, |slope|, max |step|
    Per pair   (cross):      |normalised slope A - normalised slope B|

    Also returns a per-sensor anomaly ``score`` = the largest normalised
    residual attributed to that sensor. The score drives the abstention gate
    and the ``reason`` string; it is NOT produced by the network.
    """

    def __init__(self) -> None:
        super().__init__()
        std_s, slope_s, step_s = [], [], []
        for name in SENSORS:
            a, b, c = _SCALES[name]
            std_s.append(a)
            slope_s.append(b)
            step_s.append(c)
        self.register_buffer("scale_std", torch.tensor(std_s, dtype=torch.float32))
        self.register_buffer("scale_slope", torch.tensor(slope_s, dtype=torch.float32))
        self.register_buffer("scale_step", torch.tensor(step_s, dtype=torch.float32))
        self._pairs: Tuple[Tuple[int, int], ...] = tuple(
            (SENSORS.index(a), SENSORS.index(b)) for a, b in _CROSS
        )

    @property
    def n_pairs(self) -> int:
        return len(self._pairs)

    @staticmethod
    def _slope(x: Tensor) -> Tensor:
        """Least-squares slope over the time axis. x: [B, T, C] -> [B, C]."""
        t = torch.arange(x.shape[1], dtype=x.dtype, device=x.device)
        t = t - t.mean()
        denom = (t * t).sum()
        return (x * t[:, None]).sum(dim=1) / denom

    def forward(self, window: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """window [B, T, C] -> (features [B, F], scores [B, C],
        n_std [B, C], n_slope [B, C], n_step [B, C])"""
        if window.dim() != 3:
            raise ValueError(f"expected [B,T,C], got shape {tuple(window.shape)}")
        if window.shape[1] < 2:
            raise ValueError("need at least 2 frames to compute residuals")

        n_std = window.std(dim=1) / self.scale_std
        n_slope = self._slope(window).abs() / self.scale_slope
        n_step = (window[:, 1:, :] - window[:, :-1, :]).abs().amax(dim=1) / self.scale_step

        temporal = torch.cat([n_std, n_slope, n_step], dim=-1)  # [B, 3C]

        cross_list = [
            (n_slope[:, a] - n_slope[:, b]).abs() for a, b in self._pairs
        ]
        cross = (
            torch.stack(cross_list, dim=-1) if cross_list else window.new_zeros((window.shape[0], 0))
        )  # [B, K]

        features = torch.cat([temporal, cross], dim=-1)  # [B, F]

        # per-sensor score = worst residual attributed to it
        score = torch.maximum(n_std, torch.maximum(n_slope, n_step))
        for k, (a, b) in enumerate(self._pairs):
            score[:, a] = torch.maximum(score[:, a], cross[:, k])
            score[:, b] = torch.maximum(score[:, b], cross[:, k])

        return features, score, n_std, n_slope, n_step


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class FaultDetector(nn.Module):
    """Residual encoder (fixed) + classifier head (RANDOM, untrained).

    Args:
        window: expected number of frames T
        hidden: width of the single hidden layer
        threshold: abstention gate. A per-sensor score below this means
            "nothing looks wrong" and the model reports ``none`` regardless of
            what the (untrained) head would have said.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        hidden: int = 64,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.window = window
        self.threshold = threshold
        self.encoder = ResidualEncoder()
        self.hidden = hidden
        self.head = self._build_head()

    def _build_head(self) -> nn.Sequential:
        """Fresh MLP with PyTorch's DEFAULT initialisation = random weights."""
        return nn.Sequential(
            nn.Linear(N_FEATURES, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, N_CLASSES),
        )

    def init_random(self) -> None:
        """Throw away the current head and draw a new random one.

        Exists so the demo can show an untrained network's behaviour on demand.
        """
        self.head = self._build_head()

    def weight_status(self) -> str:
        """One-line honest description for the UI / log."""
        return "RANDOM (untrained) -- detection quality pending training"

    # -- inference --------------------------------------------------------
    def forward(self, window: Tensor) -> DetectorOutput:  # type: ignore[override]
        """Score one window. Accepts [T, C] or [B, T, C]; returns one output
        for a single window, or the first batch element's output if B > 1
        (the demo only ever scores one window at a time)."""
        squeeze = window.dim() == 2
        if squeeze:
            window = window.unsqueeze(0)
        if window.dim() != 3:
            raise ValueError(f"expected [T,C] or [B,T,C], got {tuple(window.shape)}")

        t_frames = window.shape[1]

        # --- warm-up: never guess from an almost-empty window -------------
        if t_frames < MIN_WARMUP:
            return self._none_output(
                reason=f"warming up: {t_frames}/{MIN_WARMUP} frames buffered",
                warmup=True,
                confidence=0.0,
            )

        feats, scores, n_std, n_slope, n_step = self.encoder(window)
        logits = self.head(feats)  # [B, 8] -- RANDOM WEIGHTS
        probs = torch.softmax(logits, dim=-1).detach()

        sample = 0
        score_row = scores[sample].detach()
        max_score = float(score_row.max())

        # --- abstention gate (deterministic, NOT the network) -------------
        # Only applied in eval mode: during training the head must see every
        # class, otherwise "none" never gets a gradient.
        if (not self.training) and max_score < self.threshold:
            frac = max_score / self.threshold if self.threshold > 0 else 0.0
            return self._none_output(
                reason=(
                    f"all consistency residuals below threshold "
                    f"(max {max_score:.2f} < {self.threshold:.2f})"
                ),
                warmup=False,
                confidence=1.0 - frac,
            )

        # --- the (untrained) classifier decides ---------------------------
        prob_map = {cls: float(probs[sample, i]) for i, cls in enumerate(CLASSES)}
        top = max(prob_map, key=prob_map.get)  # type: ignore[arg-type]
        confidence = prob_map[top]
        reason = self._explain(top, score_row, n_std[sample], n_slope[sample], n_step[sample])

        return DetectorOutput(
            probs=prob_map,
            top_sensor=top,
            confidence=confidence,
            reason=reason,
            warmup=False,
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _none_output(reason: str, warmup: bool, confidence: float) -> DetectorOutput:
        probs = {cls: 0.0 for cls in SENSORS}
        probs["none"] = 1.0
        return DetectorOutput(
            probs=probs,
            top_sensor="none",
            confidence=confidence,
            reason=reason,
            warmup=warmup,
        )

    @staticmethod
    def _explain(
        sensor: str,
        score_row: Tensor,
        n_std: Tensor,
        n_slope: Tensor,
        n_step: Tensor,
    ) -> str:
        """Build the reason string from the residuals that actually fired.

        Plan example: "pitot frozen: airspeed variance~0 while GPS groundspeed=42 m/s"
        -- same idea, generated generically from whatever the numbers say.
        """
        if sensor == "none":
            return "classifier selected 'none'"

        j = SENSORS.index(sensor)
        score = float(score_row[j])
        std, slope, step = float(n_std[j]), float(n_slope[j]), float(n_step[j])

        # what is this sensor doing, in plain terms?
        if std < 0.15 and step < 0.15:
            behaviour = "reading is essentially flat (possible freeze/stuck)"
        elif slope >= 0.5:
            behaviour = f"reading is ramping at {slope:.2f}x scale (possible drift)"
        elif step >= 1.0:
            behaviour = f"reading jumps {step:.2f}x scale (possible spike/noise)"
        else:
            behaviour = f"variability {std:.2f}x scale"

        partners = [f"{a}/{b}" for a, b in _CROSS if sensor in (a, b)]
        partner_note = (
            f"; disagrees with {', '.join(partners)}" if partners else ""
        )
        verdict = "EXCEEDS" if score >= 1.0 else "below"
        return (
            f"{sensor}: {behaviour}; residual {score:.2f}x scale "
            f"[{verdict} 1.0]{partner_note}"
        )


# ---------------------------------------------------------------------------
# window buffering (Phase 4 will need this)
# ---------------------------------------------------------------------------
class WindowBuffer:
    """Accumulates channel dicts into the ``[T, C]`` tensor the model wants.

    Frames arrive one at a time from ``fg_reader``; the model wants a window.
    Keeps only the newest ``size`` frames.
    """

    def __init__(self, size: int = DEFAULT_WINDOW, channels: Sequence[str] = SENSORS) -> None:
        self.size = size
        self.channels = tuple(channels)
        self._buf: List[List[float]] = []

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def ready(self) -> bool:
        """True once the window holds enough frames to answer at all."""
        return len(self._buf) >= MIN_WARMUP

    @property
    def full(self) -> bool:
        return len(self._buf) >= self.size

    def push(self, channel_values: Dict[str, float]) -> None:
        """Append one frame. Missing channels are a programming error, not a
        sensor fault -- fail loudly."""
        missing = [c for c in self.channels if c not in channel_values]
        if missing:
            raise KeyError(f"frame missing channels: {missing}")
        self._buf.append([float(channel_values[c]) for c in self.channels])
        if len(self._buf) > self.size:
            del self._buf[0 : len(self._buf) - self.size]

    def tensor(self) -> Optional[Tensor]:
        """Current window as ``[T, C]``, or None while still empty."""
        if not self._buf:
            return None
        return torch.tensor(self._buf, dtype=torch.float32)

    def clear(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _demo() -> int:
    torch.manual_seed(0)
    model = FaultDetector()
    model.eval()

    print(f"FaultDetector  features={N_FEATURES}  classes={N_CLASSES}  "
          f"window={DEFAULT_WINDOW}")
    print(f"weights: {model.weight_status()}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params} (all in the classifier head)\n")

    # 1. warm-up
    buf = WindowBuffer()
    for i in range(MIN_WARMUP - 1):
        buf.push({c: 0.0 for c in SENSORS})
    out = model(buf.tensor())
    print(f"warm-up   -> {out.top_sensor:6s} conf={out.confidence:.2f}  {out.reason}")
    if out.top_sensor != "none" or not out.warmup:
        print("FAIL: warm-up must return none/warmup")
        return 1

    # 2. clean, steady data -> abstain
    buf.clear()
    for i in range(DEFAULT_WINDOW):
        buf.push({c: 0.0 for c in SENSORS})
    buf._buf[-1][SENSORS.index("accel")] = 9.81  # parked, reading 1 g
    for row in buf._buf:
        row[SENSORS.index("accel")] = 9.81
    out = model(buf.tensor())
    print(f"clean     -> {out.top_sensor:6s} conf={out.confidence:.2f}  {out.reason}")
    if out.top_sensor != "none":
        print("FAIL: clean data must abstain")
        return 1

    # 3. a real fault pattern: gps noise (big, obvious)
    buf.clear()
    g = torch.Generator().manual_seed(1)
    for i in range(DEFAULT_WINDOW):
        vals = {c: 0.0 for c in SENSORS}
        vals["accel"] = 9.81
        vals["gps"] = float(torch.randn(1, generator=g)) * 4.0  # sigma=4 m/s
        buf.push(vals)
    out = model(buf.tensor())
    print(f"gps noise -> {out.top_sensor:6s} conf={out.confidence:.2f}  {out.reason}")
    print(f"           probs={{{', '.join(f'{k}:{v:.2f}' for k, v in out.probs.items())}}}")

    # 4. gyro drift (ramp)
    buf.clear()
    for i in range(DEFAULT_WINDOW):
        vals = {c: 0.0 for c in SENSORS}
        vals["accel"] = 9.81
        vals["gyro"] = 0.01 * i * 0.045  # ~0.01 rad/s^2 at 22 Hz
        buf.push(vals)
    out = model(buf.tensor())
    print(f"gyro drift-> {out.top_sensor:6s} conf={out.confidence:.2f}  {out.reason}")

    print("\nNOTE: with random weights the CLASSIFIER's pick is not trustworthy.")
    print("      The gate and warm-up are deterministic and do work.")
    print("OK (architecture runs end to end)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
