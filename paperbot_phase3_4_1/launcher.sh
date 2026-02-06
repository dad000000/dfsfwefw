#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

python3 -m venv .venv || true
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting server on http://$HOST:$PORT ..."
python -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT"

