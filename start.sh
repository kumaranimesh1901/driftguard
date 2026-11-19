#!/usr/bin/env bash
# Start DriftGuard: databases (Docker), FastAPI backend and the Streamlit dashboard.
#
#   ./start.sh                 start everything and open the dashboard
#   NO_BROWSER=1 ./start.sh    start without opening a browser
#
# Logs: logs/api.log, logs/dashboard.log   Stop with: ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
RUN_DIR=".run"
LOG_DIR="logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

step() { printf '\033[1;34m▸\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

is_running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
port_busy()  { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

wait_for_url() {  # url seconds pidfile logfile
  local url=$1 seconds=$2 pidfile=$3 log=$4
  for _ in $(seq 1 $((seconds * 2))); do
    curl -sf "$url" >/dev/null 2>&1 && return 0
    if ! is_running "$pidfile"; then
      tail -n 25 "$log" >&2
      fail "Process exited during startup (see $log)."
    fi
    sleep 0.5
  done
  fail "Timed out waiting for $url (see $log)."
}

# 1. Dependencies -------------------------------------------------------------
step "Checking dependencies"
command -v docker >/dev/null || fail "Docker is required (install Docker Desktop)."
docker info >/dev/null 2>&1 || fail "Docker is installed but not running. Start Docker Desktop and retry."
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required ('docker compose')."
command -v curl >/dev/null || fail "curl is required."
PYTHON=""
for candidate in python3.11 python3.12 python3.13 python3; do
  if command -v "$candidate" >/dev/null &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PYTHON="$candidate"; break
  fi
done
[ -n "$PYTHON" ] || fail "Python 3.11 or newer is required."
ok "Docker, Python ($("$PYTHON" --version 2>&1)) available"

# 2. Configuration ------------------------------------------------------------
if [ ! -f .env ]; then
  cp .env.example .env
  key=$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')
  sed -i.bak "s/^SECRET_KEY=.*/SECRET_KEY=$key/" .env && rm -f .env.bak
  ok "Created .env from .env.example with a random SECRET_KEY"
fi

# 3. Python environment -------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  step "Creating virtual environment (.venv)"
  "$PYTHON" -m venv .venv
fi
req_hash=$(cat requirements.txt | shasum | cut -d' ' -f1)
if [ "$(cat "$RUN_DIR/requirements.sha" 2>/dev/null)" != "$req_hash" ]; then
  step "Installing Python dependencies (first run can take a few minutes)"
  .venv/bin/pip install -q --upgrade pip >/dev/null
  .venv/bin/pip install -q -r requirements.txt
  echo "$req_hash" > "$RUN_DIR/requirements.sha"
fi
ok "Python environment ready"

# 4. Databases ----------------------------------------------------------------
step "Starting PostgreSQL and MongoDB (Docker)"
docker compose up -d postgres mongo >/dev/null 2>&1 || fail "docker compose could not start the databases."
for service in postgres mongo; do
  for _ in $(seq 1 60); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q "$service")" 2>/dev/null || true)
    [ "$status" = "healthy" ] && break
    sleep 1
  done
  [ "$status" = "healthy" ] || fail "$service did not become healthy (docker compose logs $service)."
done
ok "Databases healthy"

# 5. FastAPI backend ----------------------------------------------------------
if is_running "$RUN_DIR/api.pid"; then
  ok "Backend already running (pid $(cat "$RUN_DIR/api.pid"))"
else
  port_busy "$API_PORT" && fail "Port $API_PORT is in use by another program. Free it or set API_PORT."
  step "Starting backend on port $API_PORT"
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" \
    > "$LOG_DIR/api.log" 2>&1 &
  echo $! > "$RUN_DIR/api.pid"
  wait_for_url "http://127.0.0.1:$API_PORT/health" 60 "$RUN_DIR/api.pid" "$LOG_DIR/api.log"
  ok "Backend running"
fi

# 6. Streamlit dashboard ------------------------------------------------------
if is_running "$RUN_DIR/dashboard.pid"; then
  ok "Dashboard already running (pid $(cat "$RUN_DIR/dashboard.pid"))"
else
  port_busy "$UI_PORT" && fail "Port $UI_PORT is in use by another program. Free it or set UI_PORT."
  step "Starting dashboard on port $UI_PORT"
  DRIFTGUARD_API="http://127.0.0.1:$API_PORT" nohup .venv/bin/streamlit run frontend/dashboard.py \
    --server.address 127.0.0.1 --server.port "$UI_PORT" --server.headless true \
    --server.maxUploadSize "$(grep -E '^MAX_UPLOAD_MB=' .env | cut -d= -f2 || echo 50)" \
    --browser.gatherUsageStats false > "$LOG_DIR/dashboard.log" 2>&1 &
  echo $! > "$RUN_DIR/dashboard.pid"
  wait_for_url "http://127.0.0.1:$UI_PORT/_stcore/health" 60 "$RUN_DIR/dashboard.pid" "$LOG_DIR/dashboard.log"
  ok "Dashboard running"
fi

URL="http://localhost:$UI_PORT"
printf '\n\033[1mDriftGuard is ready:\033[0m %s\n' "$URL"
printf 'Stop with ./stop.sh   ·   logs in %s/\n' "$LOG_DIR"
if [ "${NO_BROWSER:-0}" != "1" ]; then
  if command -v open >/dev/null; then open "$URL"
  elif command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 || true
  fi
fi
