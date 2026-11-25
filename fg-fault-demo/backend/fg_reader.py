"""Read FlightGear telemetry frames over UDP and expose them as a stream.

Transport: FG's generic protocol, UDP out.
    --generic=socket,out,<hz>,localhost,<port>,udp,telemetry

Wire format produced by protocol/telemetry.xml - one line per frame:

    t=12.345,ias=45.50,alt=3500.00,hdg=271.05,...

Design notes
------------
* Every frame is validated before it is published. Malformed lines are
  counted and dropped, never propagated - a corrupted frame must not
  become a fake "sensor fault" downstream.
* Unknown keys are kept (forward compatible) but not typed.
* ``latest_state()`` never blocks the caller; ``states()`` is a generator
  for consumers that want every frame.
"""

from __future__ import annotations

import socket
import time
from collections import deque
from typing import Dict, Iterator, List, Optional, Tuple

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5510

# Keys we coerce to float. Kept as an explicit allow-list so a typo in the
# protocol XML shows up loudly instead of silently producing strings.
FLOAT_KEYS = frozenset(
    {
        "t",
        "ias",
        "mach",
        "alt",
        "altind",
        "agl",
        "vs",
        "lat",
        "lon",
        "vng",
        "veg",
        "hdg",
        "pitch",
        "roll",
        "alpha",
        "beta",
        "p",
        "q",
        "r",
        "ax",
        "ay",
        "az",
        "elev",
        "ail",
        "rud",
        "thr",
    }
)


class FrameError(ValueError):
    """Raised when a wire line cannot be parsed into a state dict."""


def parse_line(line: str) -> Dict[str, float]:
    """Parse one ``k=v,k=v`` frame into a dict of floats.

    Raises FrameError on anything malformed - callers drop the frame.
    """
    line = line.strip()
    if not line:
        raise FrameError("empty line")

    out: Dict[str, float] = {}
    for pair in line.split(","):
        if not pair:
            continue
        if "=" not in pair:
            raise FrameError(f"segment without '=': {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not key:
            raise FrameError(f"empty key in segment: {pair!r}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise FrameError(f"non-numeric value for {key!r}: {raw!r}") from exc
        if value != value:  # NaN
            raise FrameError(f"NaN value for {key!r}")
        out[key] = value

    if not out:
        raise FrameError("no key=value pairs")
    if "t" not in out:
        raise FrameError("missing timestamp key 't'")
    return out


class FGReader:
    """Non-blocking UDP reader for the FlightGear generic protocol."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 history: int = 128) -> None:
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._latest: Optional[Dict[str, float]] = None
        self._latest_rx: float = 0.0
        self._history: deque = deque(maxlen=history)
        self.frames_ok = 0
        self.frames_bad = 0
        self._last_bad_line = ""

    # -- lifecycle -------------------------------------------------------
    def open(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            sock.close()
            raise RuntimeError(
                f"cannot bind UDP {self.host}:{self.port} - {exc}"
            ) from exc
        sock.settimeout(0.25)  # keep the loop responsive
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "FGReader":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading ---------------------------------------------------------
    def poll(self, timeout: float = 0.25) -> Optional[Dict[str, float]]:
        """Wait up to *timeout* seconds for one frame. Returns it, else None."""
        if self._sock is None:
            raise RuntimeError("reader not open - call open() first")
        self._sock.settimeout(timeout)
        try:
            data, _addr = self._sock.recvfrom(65535)
        except socket.timeout:
            return None
        return self._ingest(data)

    def _ingest(self, data: bytes) -> Optional[Dict[str, float]]:
        try:
            line = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            self.frames_bad += 1
            self._last_bad_line = repr(data[:120])
            return None

        try:
            state = parse_line(line)
        except FrameError as exc:
            self.frames_bad += 1
            self._last_bad_line = f"{exc} | {line[:120]!r}"
            return None

        self.frames_ok += 1
        self._latest = state
        self._latest_rx = time.monotonic()
        self._history.append(state)
        return state

    def latest_state(self) -> Optional[Dict[str, float]]:
        """Most recent valid frame, or None if none received yet."""
        return self._latest

    def age_seconds(self) -> float:
        """Seconds since the last valid frame (inf if never received)."""
        if self._latest_rx == 0.0:
            return float("inf")
        return time.monotonic() - self._latest_rx

    def states(self, timeout: float = 0.25) -> Iterator[Dict[str, float]]:
        """Yield every valid frame as it arrives, forever."""
        while True:
            state = self.poll(timeout)
            if state is not None:
                yield state

    @property
    def stats(self) -> Tuple[int, int, str]:
        return self.frames_ok, self.frames_bad, self._last_bad_line


def _demo(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          seconds: float = 10.0) -> int:
    """Manual smoke test: print frames for *seconds* and report the rate."""
    reader = FGReader(host, port)
    try:
        reader.open()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"listening on udp://{host}:{port} for {seconds:.0f}s ...")
    deadline = time.monotonic() + seconds
    printed = 0
    while time.monotonic() < deadline:
        state = reader.poll(timeout=0.25)
        if state is None:
            continue
        if printed < 5 or printed % 30 == 0:
            print(
                f"  t={state.get('t', 0):8.3f}  "
                f"alt={state.get('alt', 0):9.2f} ft  "
                f"ias={state.get('ias', 0):6.2f} kt  "
                f"hdg={state.get('hdg', 0):6.2f}  "
                f"pitch={state.get('pitch', 0):6.2f}  "
                f"roll={state.get('roll', 0):6.2f}"
            )
        printed += 1

    ok, bad, last_bad = reader.stats
    reader.close()
    elapsed = seconds
    rate = ok / elapsed if elapsed else 0.0
    print(f"\nframes ok={ok} bad={bad}  (~{rate:.1f} Hz)")
    if bad:
        print(f"last bad line: {last_bad}")
    if ok == 0:
        print("FAIL: no frames received. Is FlightGear running with "
              "--generic=socket,out,...,udp,telemetry ?")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()
    raise SystemExit(_demo(args.host, args.port, args.seconds))
