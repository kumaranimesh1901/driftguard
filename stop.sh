#!/usr/bin/env bash
# Stop DriftGuard cleanly. Stored data is never deleted: database containers are
# stopped (not removed) and their volumes are kept.
#
#   ./stop.sh            stop dashboard, backend and database containers
#   ./stop.sh --keep-db  stop dashboard and backend, leave databases running
set -uo pipefail
cd "$(dirname "$0")"
RUN_DIR=".run"

ok() { printf '\033[1;32m✓\033[0m %s\n' "$*"; }

stop_process() {  # name pidfile
  local name=$1 pidfile=$2 pid
  if [ ! -f "$pidfile" ]; then
    ok "$name not running"
    return
  fi
  pid=$(cat "$pidfile")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
    ok "$name stopped"
  else
    ok "$name not running"
  fi
  rm -f "$pidfile"
}

stop_process "Dashboard" "$RUN_DIR/dashboard.pid"
stop_process "Backend" "$RUN_DIR/api.pid"

if [ "${1:-}" = "--keep-db" ]; then
  ok "Databases left running"
elif command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker compose stop postgres mongo >/dev/null 2>&1 && ok "Databases stopped (data volumes kept)"
fi
