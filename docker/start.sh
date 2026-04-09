#!/usr/bin/env bash
set -euo pipefail

gunicorn \
  --bind "${GUNICORN_BIND}" \
  --workers "${GUNICORN_WORKERS}" \
  --threads "${GUNICORN_THREADS}" \
  --timeout "${GUNICORN_TIMEOUT}" \
  "app:server" &
gunicorn_pid=$!

nginx -g "daemon off;" &
nginx_pid=$!

shutdown() {
  kill -TERM "${gunicorn_pid}" "${nginx_pid}" 2>/dev/null || true
  wait "${gunicorn_pid}" "${nginx_pid}" 2>/dev/null || true
}

trap shutdown INT TERM

wait -n "${gunicorn_pid}" "${nginx_pid}"
exit_code=$?
shutdown
exit "${exit_code}"
