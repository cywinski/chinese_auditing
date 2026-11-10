#!/bin/bash
# ABOUTME: Runs E2 generation for a sequence of vLLM servers: start server, run its variants, stop server.
# ABOUTME: Usage: rebuttal_e2_run_servers.sh <config> <run_dir> "<driver overrides>" <server> [<server> ...]
set -euo pipefail

CONFIG="$1"; RUN_DIR="$2"; OVERRIDES="$3"; shift 3
mkdir -p "$RUN_DIR/logs"

for SERVER in "$@"; do
    TS=$(date +%Y%m%d_%H%M%S)
    SERVER_LOG="$RUN_DIR/logs/vllm_${SERVER}_${TS}.log"
    echo "[$(date -u +%FT%TZ)] starting server $SERVER (log: $SERVER_LOG)"
    python src/experiments/rebuttal_e2_serve.py "$CONFIG" "$SERVER" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

    # Driver waits for the server; abort if the server process dies meanwhile.
    # shellcheck disable=SC2086
    python src/experiments/rebuttal_e2_capability.py "$CONFIG" mode=generate run_dir="$RUN_DIR" \
        server="$SERVER" $OVERRIDES 2>&1 | tee "$RUN_DIR/logs/generate_${SERVER}_${TS}.log" &
    DRIVER_PID=$!
    while kill -0 $DRIVER_PID 2>/dev/null; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "[$(date -u +%FT%TZ)] SERVER $SERVER DIED; see $SERVER_LOG"
            tail -30 "$SERVER_LOG"
            pkill -P $$ || true
            exit 1
        fi
        sleep 5
    done
    wait $DRIVER_PID

    echo "[$(date -u +%FT%TZ)] stopping server $SERVER"
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
    trap - EXIT
done
echo "[$(date -u +%FT%TZ)] ALL SERVERS DONE: $*"
