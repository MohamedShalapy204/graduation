#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh -- ONE command that brings up the whole demo.
#
#   ./fg-fault-demo/run.sh
#
# It will, in order:
#   1. preflight  - venv, web/, telemetry protocol symlink, FlightGear binary
#   2. FlightGear - start it with the lean flags IF it is not already running
#                   (never a second instance), then wait for live telemetry
#   3. backend    - start server.py IF port 8000 is free; if it is taken by
#                   anything else, say so plainly and exit non-zero
#   4. browser    - open http://localhost:8000/
#   5. follow     - tail the server log so failures are visible, not silent
#
# Options:
#   --no-fg         do not start FlightGear (assume it is already running)
#   --no-browser    do not open a browser tab
#   --fresh         kill and restart OUR server even if it is already up
#   --port N        dashboard port         (default 8000)
#   --fg-port N     FlightGear UDP port    (default 5510)
#   -h | --help
#
# Failure modes (all print a clear message and exit 1, never hang silently):
#   missing venv / web/ / AppImage, low memory before launching FG, FG exits
#   during startup, no telemetry after launch, port held by another program.
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../fg-fault-demo
ROOT="$(cd "$HERE/.." && pwd)"                        # repo root
PY="$ROOT/.venv/bin/python"
FG_BIN="$ROOT/flightgear-2024.1.7-linux-amd64.AppImage"
PROTOCOL_XML="$HERE/protocol/telemetry.xml"
FG_ROOT_DIR="$HOME/.fgfs/fgdata_2024_1"
BACKEND="$HERE/backend"
WEB="$HERE/web"
LOG_DIR="${LOG_DIR:-/tmp/fg-fault-demo}"
FG_LOG="$LOG_DIR/flightgear.log"
SRV_LOG="$LOG_DIR/server.log"

PORT=8000
FG_PORT=5510
WANT_FG=1
WANT_BROWSER=1
FRESH=0
FOLLOW=1

# ---------------------------------------------------------------- helpers
say()  { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[run]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[run]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[run] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --no-fg)       WANT_FG=0 ;;
    --no-browser)  WANT_BROWSER=0 ;;
    --fresh)       FRESH=1 ;;
    --port)        PORT="${2:?--port needs a value}"; shift ;;
    --fg-port)     FG_PORT="${2:?--fg-port needs a value}"; shift ;;
    --no-follow)   FOLLOW=0 ;;
    -h|--help)     usage ;;
    *) die "unknown option: $1  (try --help)" ;;
  esac
  shift
done

mkdir -p "$LOG_DIR"

# PID of whoever holds a TCP listen port, or empty
port_pid() {
  ss -tlnp 2>/dev/null | grep -E ":$1[[:space:]]" | grep -oE 'pid=[0-9]+' \
    | head -1 | cut -d= -f2
}

# Our FlightGear = the one carrying our telemetry flag (never match ourselves)
fg_pid() {
  pgrep -f "fgfs.*--generic=socket,out,30,localhost,$FG_PORT,udp,telemetry" \
    | head -1
}

# ANY FlightGear, whatever flags it was started with. Used so we can never
# accidentally launch a second instance next to one the user opened by hand.
fg_any_pid() {
  pgrep -f "[f]gfs[[:space:]]+--aircraft=" | head -1
}

# ---------------------------------------------------------------- 1. preflight
say "preflight ..."

[ -x "$PY" ] || die "python venv not found at $PY
       create it first:  cp -a ~/projects/rl-project/.venv $ROOT/.venv
       then repair its paths (see plans/01_flightgear_demo.md, Phase 0)."

[ -d "$WEB" ] || die "web/ not found at $WEB"
[ -f "$WEB/index.html" ] || die "web/index.html missing - nothing to serve"

[ -f "$PROTOCOL_XML" ] || die "protocol definition missing: $PROTOCOL_XML"

if [ ! -e "$FG_ROOT_DIR/Protocol/telemetry.xml" ]; then
  say "linking telemetry.xml into \$FG_ROOT/Protocol/ ..."
  ln -sfn "$PROTOCOL_XML" "$FG_ROOT_DIR/Protocol/telemetry.xml" \
    || die "could not link into $FG_ROOT_DIR/Protocol/
       (FlightGear would then not find the 'telemetry' protocol)"
fi

# quick sanity: the modules import cleanly before we commit to anything
"$PY" - <<PYEOF || die "backend modules failed to import (see above)"
import sys
sys.path.insert(0, r"$BACKEND")
import fg_reader, sensor_channels, faults, model, server   # noqa: F401
print("[run] backend modules import OK")
PYEOF

# ---------------------------------------------------------------- 2. FlightGear
if [ "$WANT_FG" = 1 ]; then
  if pid="$(fg_pid)" && [ -n "$pid" ]; then
    ok "FlightGear already running (pid $pid) - reusing it, not starting a second one"
  elif other="$(fg_any_pid)" && [ -n "$other" ]; then
    die "FlightGear is already running (pid $other) but was NOT started with our
       telemetry flag, so there is nothing to read.
       Close that instance first, then re-run this script - it will start
       FlightGear with the correct flags. (Never two at once: it needs ~2 GB.)"
  else
    [ -f "$FG_BIN" ] || die "FlightGear AppImage not found: $FG_BIN"

    avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    avail_mb=$(( avail_kb / 1024 ))
    if [ "$avail_mb" -lt 1200 ]; then
      warn "only ${avail_mb} MB RAM available; FlightGear needs ~2 GB."
      warn "it was OOM-killed before. Close a browser tab or two, then re-run."
      die "not launching FlightGear with insufficient memory"
    fi

    say "starting FlightGear (c172p, lean flags, telemetry -> udp://localhost:$FG_PORT) ..."
    : > "$FG_LOG"
    ( cd "$ROOT" && exec nohup "$FG_BIN" \
        --aircraft=c172p --launcher=0 --terrasync=0 --sound=0 --splash-screen=0 \
        --ai-models=0 --ai-traffic=0 \
        --random-vegetation=0 --random-objects=0 --random-buildings=0 \
        --lod-levels=2 --texture-filtering=1 --max-fps=30 --timeofday=noon \
        --generic=file,out,30,"$LOG_DIR"/fg_phase.txt,telemetry \
        --generic=socket,out,30,localhost,"$FG_PORT",udp,telemetry \
        >> "$FG_LOG" 2>&1 ) &
    FG_LAUNCHER=$!

    # give it a moment, then confirm it actually stayed alive
    sleep 6
    if ! pid="$(fg_pid)" || [ -z "$pid" ]; then
      warn "FlightGear exited during startup. Last lines of $FG_LOG:"
      tail -n 25 "$FG_LOG" >&2 || true
      die "could not start FlightGear - fix the error above and re-run"
    fi
    ok "FlightGear up (pid $pid)"
  fi

  # wait for a real telemetry frame so the dashboard is never born blind
  say "waiting for telemetry on udp://localhost:$FG_PORT ..."
  if ! "$PY" "$BACKEND/fg_reader.py" --seconds 12 --port "$FG_PORT" >"$LOG_DIR/telemetry_probe.log" 2>&1; then
    warn "no telemetry after 12 s. Probe output:"
    tail -n 20 "$LOG_DIR/telemetry_probe.log" >&2 || true
    die "FlightGear is running but not sending.
       It must be started with:
         --generic=socket,out,30,localhost,$FG_PORT,udp,telemetry
       and the protocol file must exist at:
         $FG_ROOT_DIR/Protocol/telemetry.xml"
  fi
  ok "telemetry flowing"
else
  say "--no-fg: assuming FlightGear is already running"
  if ! pid="$(fg_pid)" || [ -z "$pid" ]; then
    warn "no process found with our telemetry flag; the dashboard may show NO SIGNAL."
  fi
fi

# ---------------------------------------------------------------- 3. backend
existing="$(port_pid "$PORT")"
if [ -n "$existing" ]; then
  cmdline=$(tr '\0' ' ' < "/proc/$existing/cmdline" 2>/dev/null || true)
  if [ "$FRESH" = 1 ]; then
    say "--fresh: stopping our server (pid $existing)"
    kill "$existing" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      [ -n "$(port_pid "$PORT")" ] || break
      sleep 0.5
    done
    [ -n "$(port_pid "$PORT")" ] && die "could not free port $PORT (pid $existing)"
  elif printf '%s' "$cmdline" | grep -q "server.py"; then
    ok "backend already running on port $PORT (pid $existing) - reusing it"
    WANT_NEW_SERVER=0
  else
    die "port $PORT is already in use by another program (pid $existing):
         $cmdline
       Stop it, or pick another port:  ./run.sh --port 8001"
  fi
fi

if [ -z "$(port_pid "$PORT")" ]; then
  say "starting backend on http://localhost:$PORT/ ..."
  : > "$SRV_LOG"
  ( cd "$BACKEND" && exec "$PY" -u server.py --port "$PORT" --fg-port "$FG_PORT" \
      >> "$SRV_LOG" 2>&1 ) &
  SRV_PID=$!

  for _ in $(seq 1 40); do
    [ -n "$(port_pid "$PORT")" ] && break
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      warn "backend died during startup. Log:"
      tail -n 30 "$SRV_LOG" >&2 || true
      die "could not start backend (see log above: $SRV_LOG)"
    fi
    sleep 0.25
  done
  [ -n "$(port_pid "$PORT")" ] || die "backend did not bind port $PORT in 10 s (log: $SRV_LOG)"
  ok "backend up (pid $(port_pid "$PORT"))"
fi

# ---------------------------------------------------------------- 4. browser
if [ "$WANT_BROWSER" = 1 ]; then
  say "opening http://localhost:$PORT/ ..."
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$PORT/" >/dev/null 2>&1 \
      || warn "could not open a browser automatically - go to http://localhost:$PORT/ yourself"
  else
    warn "xdg-open not found - open http://localhost:$PORT/ manually"
  fi
fi

# ---------------------------------------------------------------- 5. follow
ok "demo ready  ->  http://localhost:$PORT/"
say "fault buttons: Pitot Blockage · Gyro Drift · Baro Bias · Clear"
say "server log: $SRV_LOG      flightgear log: $FG_LOG"
say "Ctrl-C stops the backend (FlightGear keeps running)."

if [ "$FOLLOW" = 1 ]; then
  trap 'printf "\n[run] stopping backend\n"; exit 0' INT TERM
  tail -n 0 -f "$SRV_LOG"
fi
