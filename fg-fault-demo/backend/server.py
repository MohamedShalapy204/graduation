"""Phase 4 -- WebSocket server + static host for the dashboard.

One process, ONE port: it serves ``web/`` over plain HTTP and upgrades
``/ws`` to a WebSocket. Opening ``http://localhost:8000`` is enough.

Pipeline, once per FlightGear frame:

    fg_reader  ->  sensor_channels.build_channels  ->  faults.apply
              ->  WindowBuffer  ->  FaultDetector  ->  broadcast JSON

Message contract (plan section 3, Phase 4). Deliberately shaped so the
eventual .NET/SignalR + React stack can replace this file and ``web/``
without touching the AI::

    -> client (on connect)
        {"type":"hello", "channels":[...], "units":{...},
         "faults":[...], "weights":"...", "port":8000}

    -> client (every frame)
        {"type":"telemetry", "t":..., "raw":{...26 keys...},
         "channels":{...corrupted...}, "truth":{...clean...},
         "attitude":{"pitch":..,"roll":..,"hdg":..},
         "verdict":{"top_sensor","confidence","reason","probs","warmup"},
         "active_fault": null | "pitot_blockage",
         "stats":{"fps":..,"frames_ok":..,"frames_bad":..,"age_s":..}}

    <- server
        {"cmd":"set_fault","fault":"pitot_blockage"}   # or {"cmd":"clear"}

Failure modes (Definition of Done): FlightGear down -> prints how to start it
and keeps serving the page with a visible "NO SIGNAL" banner; port in use ->
exits with a message naming the port; bad fault name -> error frame to that
client only, server keeps running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from faults import PRESETS, FaultController
from fg_reader import DEFAULT_HOST, DEFAULT_PORT, FGReader
from model import CLASSES, DEFAULT_WINDOW, FaultDetector, WindowBuffer
from sensor_channels import CHANNELS, CHANNEL_META, build_channels

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PORT = 8000  # same port -- /ws upgrades, everything else is static

# clients currently connected
CLIENTS: Set[ServerConnection] = set()


# ---------------------------------------------------------------------------
# static file serving (inside the websockets handshake path)
# ---------------------------------------------------------------------------
def _static_response(path: str) -> Response:
    """Map a URL path to a file under ``web/``. Never escapes that directory."""
    if path in ("", "/"):
        path = "/index.html"
    # strip query string / fragments if a browser added them
    path = path.split("?", 1)[0].split("#", 1)[0]
    rel = path.lstrip("/")

    root = WEB_DIR.resolve()
    try:
        target = (root / rel).resolve()
    except (OSError, ValueError):
        return Response(404, "Not Found", Headers(), b"404 not found")

    if root not in target.parents and target != root:
        return Response(403, "Forbidden", Headers(), b"403 forbidden")
    if not target.is_file():
        return Response(404, "Not Found", Headers(), b"404 not found")

    body = target.read_bytes()
    ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
    headers = Headers(
        [
            ("Content-Type", ctype),
            ("Content-Length", str(len(body))),
            ("Cache-Control", "no-store"),  # demo: always fresh
        ]
    )
    return Response(200, "OK", headers, body)


def process_request(connection: ServerConnection, request: Request) -> Optional[Response]:
    """Serve static files; return None for /ws so the handshake proceeds."""
    if request.path == "/ws":
        return None
    return _static_response(request.path)


# ---------------------------------------------------------------------------
# application state
# ---------------------------------------------------------------------------
class App:
    """Holds the reader, fault controller, window buffer and model."""

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.reader = FGReader(port=port)
        self.faults = FaultController()
        self.model = FaultDetector()
        self.model.eval()
        self.buffer = WindowBuffer(size=DEFAULT_WINDOW)

        self.frames_ok = 0
        self.fps = 0.0
        self._fps_window: list[float] = []
        self.last_state: Optional[Dict[str, float]] = None
        self.started = time.monotonic()

    # -- fault control ----------------------------------------------------
    def set_fault(self, name: str) -> None:
        """Raises ValueError with a helpful message on a bad name."""
        if name not in PRESETS:
            raise ValueError(
                f"unknown fault {name!r}; available: {', '.join(sorted(PRESETS))}"
            )
        self.faults.set_active(name)

    def clear_fault(self) -> None:
        self.faults.clear()

    # -- one frame --------------------------------------------------------
    def step(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Truth -> channels -> fault -> window -> verdict -> message dict."""
        truth = build_channels(state)
        corrupted = self.faults.apply(truth, state["t"])
        self.buffer.push(corrupted)

        if self.buffer.ready:
            tensor = self.buffer.tensor()
            verdict = self.model(tensor)
            # Deterministic part: how anomalous each sensor looks, before the
            # (currently untrained) head votes. This is what actually names the
            # faulty sensor today, so the dashboard must show it next to the
            # head's verdict or the demo looks broken.
            # NOTE: encoder.forward demands a batch dim -- only FaultDetector
            # is lenient about a bare [T, C] window.
            with torch.no_grad():
                _, score, *_ = self.model.encoder(tensor.unsqueeze(0))
            sensor_scores = {
                c: round(float(score[0, i]), 4) for i, c in enumerate(CHANNELS)
            }
            gate_open = bool(score.max() >= self.model.threshold)
        else:
            verdict = None
            sensor_scores = {c: 0.0 for c in CHANNELS}
            gate_open = False

        # fps over the last ~1 s
        now = time.monotonic()
        self._fps_window.append(now)
        cutoff = now - 1.0
        self._fps_window = [t for t in self._fps_window if t >= cutoff]
        self.fps = float(len(self._fps_window))

        active = self.faults.active
        return {
            "type": "telemetry",
            "t": state["t"],
            "raw": {k: state[k] for k in sorted(state)},
            "channels": corrupted,
            "truth": truth,
            "attitude": {
                "pitch": state.get("pitch", 0.0),
                "roll": state.get("roll", 0.0),
                "hdg": state.get("hdg", 0.0),
            },
            "verdict": (
                {
                    "top_sensor": verdict.top_sensor,
                    "confidence": verdict.confidence,
                    "reason": verdict.reason,
                    "probs": verdict.probs,
                    "warmup": verdict.warmup,
                    "sensor_scores": sensor_scores,
                    "gate_open": gate_open,
                }
                if verdict is not None
                else {
                    "top_sensor": "none",
                    "confidence": 0.0,
                    "reason": f"warming up: {len(self.buffer)}/{DEFAULT_WINDOW} frames",
                    "probs": {c: 0.0 for c in CLASSES if c != "none"} | {"none": 1.0},
                    "warmup": True,
                    "sensor_scores": sensor_scores,
                    "gate_open": gate_open,
                }
            ),
            "active_fault": active.channel if active else None,
            "active_fault_name": (
                f"{active.fault_type}({active.channel})" if active else None
            ),
            "stats": {
                "fps": round(self.fps, 1),
                "frames_ok": self.reader.frames_ok,
                "frames_bad": self.reader.frames_bad,
                "age_s": round(self.reader.age_seconds(), 3),
                "window": len(self.buffer),
            },
        }

    def hello(self) -> Dict[str, Any]:
        return {
            "type": "hello",
            "channels": list(CHANNELS),
            "units": {c: CHANNEL_META[c]["unit"] for c in CHANNELS},
            "faults": sorted(PRESETS),
            "weights": self.model.weight_status(),
            "threshold": self.model.threshold,
            "window": DEFAULT_WINDOW,
        }


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------
async def handler(websocket: ServerConnection, app: App) -> None:
    """One coroutine per connected browser tab."""
    CLIENTS.add(websocket)
    print(f"[ws] client connected ({len(CLIENTS)} total)")
    try:
        await websocket.send(json.dumps(app.hello()))
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await websocket.send(
                    json.dumps({"type": "error", "message": "bad JSON"})
                )
                continue
            if not isinstance(msg, dict):
                # a JSON *string*/number/array parses fine but carries no cmd --
                # say so rather than dropping it silently
                await websocket.send(
                    json.dumps(
                        {"type": "error", "message": "expected a JSON object"}
                    )
                )
                continue
            cmd = msg.get("cmd")
            try:
                if cmd == "set_fault":
                    app.set_fault(str(msg.get("fault", "")))
                    print(f"[fault] ON  -> {app.faults.describe()}")
                elif cmd == "clear":
                    app.clear_fault()
                    print("[fault] OFF -> clean")
                else:
                    await websocket.send(
                        json.dumps(
                            {"type": "error", "message": f"unknown cmd {cmd!r}"}
                        )
                    )
                    continue
            except ValueError as exc:
                await websocket.send(
                    json.dumps({"type": "error", "message": str(exc)})
                )
                continue
            # acknowledge so the UI can light the right button
            await websocket.send(
                json.dumps(
                    {
                        "type": "fault_state",
                        "active_fault": (
                            app.faults.active.channel if app.faults.active else None
                        ),
                        "describe": app.faults.describe(),
                    }
                )
            )
    except Exception as exc:  # client vanished / protocol error
        print(f"[ws] client error: {exc}")
    finally:
        CLIENTS.discard(websocket)
        print(f"[ws] client disconnected ({len(CLIENTS)} total)")


async def broadcast(payload: Dict[str, Any]) -> None:
    if not CLIENTS:
        return
    text = json.dumps(payload)
    dead = []
    for ws in list(CLIENTS):
        try:
            await ws.send(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        CLIENTS.discard(ws)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
async def telemetry_loop(app: App) -> None:
    """Poll FlightGear forever; broadcast one message per frame."""
    print(f"[fg] listening on udp://{DEFAULT_HOST}:{app.reader.port} ...")
    idle_since = time.monotonic()
    announced = False

    while True:
        # timeout is short so the event loop stays responsive to WS traffic
        try:
            state = app.reader.poll(timeout=0.01)
        except RuntimeError as exc:  # reader not open
            print(f"[fg] {exc}")
            await asyncio.sleep(1.0)
            continue

        if state is None:
            age = app.reader.age_seconds()
            if age > 3.0 and not announced:
                announced = True
                idle_since = time.monotonic()
                # age is inf until the very first frame arrives
                how = "never received any" if age == float("inf") else f"none for {age:.1f}s"
                print(
                    "[fg] NO SIGNAL (%s) - is FlightGear running with\n"
                    "     --generic=socket,out,30,localhost,%d,udp,telemetry ?"
                    % (how, app.reader.port)
                )
                await broadcast(
                    {
                        "type": "status",
                        "ok": False,
                        "message": (
                            f"No telemetry from FlightGear (udp {app.reader.port}): {how}"
                        ),
                    }
                )
            await asyncio.sleep(0.02)
            continue

        if announced:
            announced = False
            print("[fg] signal restored")
            await broadcast({"type": "status", "ok": True, "message": "telemetry live"})

        app.frames_ok += 1
        app.last_state = state
        try:
            msg = app.step(state)
        except ValueError as exc:
            # a frame missing keys is a protocol bug, not a sensor fault
            print(f"[fg] bad frame: {exc}")
            await asyncio.sleep(0.05)
            continue
        await broadcast(msg)


async def run(port: int, fg_port: int) -> int:
    app = App(port=fg_port)
    try:
        app.reader.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    # the handler is a closure over `app` -- websockets' Request object has no
    # extension bag to stash it in
    async def _handler(websocket: ServerConnection) -> None:
        await handler(websocket, app)

    try:
        async with serve(
            _handler,
            "0.0.0.0",
            port,
            process_request=process_request,
            ping_interval=20,
            ping_timeout=20,
        ) as server:
            print(f"dashboard: http://localhost:{port}/   (WebSocket on /ws)")
            print(f"model    : {app.model.weight_status()}")
            print(f"faults   : {', '.join(sorted(PRESETS))}")
            print("Ctrl-C to stop.\n")
            await telemetry_loop(app)
    except OSError as exc:
        print(f"ERROR: cannot bind port {port} - {exc}")
        print("       (is another instance of server.py already running?)")
        app.reader.close()
        return 1
    except asyncio.CancelledError:
        pass
    finally:
        app.reader.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT,
                    help="HTTP + WebSocket port (default 8000)")
    ap.add_argument("--fg-port", type=int, default=DEFAULT_PORT,
                    help="UDP port FlightGear sends telemetry to")
    args = ap.parse_args()
    if not WEB_DIR.is_dir():
        print(f"ERROR: web/ directory not found at {WEB_DIR}")
        return 1
    try:
        return asyncio.run(run(args.port, args.fg_port))
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
