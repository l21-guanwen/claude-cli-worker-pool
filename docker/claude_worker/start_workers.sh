#!/bin/sh
# Starts WORKERS_PER_ACCOUNT uvicorn processes on consecutive ports from BASE_PORT.
# Each process shares CLAUDE_CONFIG_DIR — sessions are UUID-named files, no conflicts.
#
# Usage: set WORKERS_PER_ACCOUNT (default 1) and BASE_PORT (default 8000).
# The pool service must know these ports: ACCOUNT_BASE_URLS × WORKERS_PER_ACCOUNT.

WORKERS=${WORKERS_PER_ACCOUNT:-1}
BASE=${BASE_PORT:-8000}
pids=""

for i in $(seq 0 $((WORKERS - 1))); do
  PORT=$((BASE + i))
  uvicorn worker_api:app --host 0.0.0.0 --port "$PORT" --log-level warning &
  pids="$pids $!"
  echo "Started worker on :$PORT (CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR:-default})"
done

echo "All $WORKERS worker(s) started. Waiting..."

# Exit if any worker exits (keeps container alive only while all workers run)
wait $pids
