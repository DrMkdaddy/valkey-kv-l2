#!/usr/bin/env bash
# Run a command with a Valkey on $VALKEY_PORT, starting a throwaway one when
# nothing is listening. Stop the server again only if this script started it.
set -euo pipefail

port="${VALKEY_PORT:-6390}"

if ! command -v valkey-server >/dev/null 2>&1; then
  echo "valkey-server not on PATH. Run this from: nix develop" >&2
  exit 1
fi

started=0
if ! valkey-cli -p "$port" ping >/dev/null 2>&1; then
  valkey-server --port "$port" --save '' --appendonly no --daemonize yes \
    --logfile "/tmp/valkey-kv-l2-${port}.log"
  started=1
  for _ in $(seq 1 100); do
    if valkey-cli -p "$port" ping >/dev/null 2>&1; then break; fi
    sleep 0.1
  done
  if ! valkey-cli -p "$port" ping >/dev/null 2>&1; then
    echo "valkey did not start on port $port; see /tmp/valkey-kv-l2-${port}.log" >&2
    exit 1
  fi
fi

cleanup() {
  if [ "$started" = 1 ]; then
    valkey-cli -p "$port" shutdown nosave >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

"$@"
