#!/usr/bin/env bash
# Startet das komplette Webtool: Web-App (Port 8000) + ML-Server (Port 8001).
# Beim ersten Aufruf wird die Python-Umgebung eingerichtet (lädt u. a. PyTorch).
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null; then
  echo "Hinweis: ffmpeg fehlt (für die ML-Analyse nötig): sudo apt install ffmpeg" >&2
fi

if [ ! -d diarization-server/.venv ]; then
  echo "Richte Python-Umgebung ein (einmalig, kann einige Minuten dauern) …"
  python3 -m venv diarization-server/.venv
  diarization-server/.venv/bin/pip install -r diarization-server/requirements.txt
fi

diarization-server/.venv/bin/uvicorn --app-dir diarization-server server:app --port 8001 &
ML_PID=$!
trap 'kill $ML_PID 2>/dev/null' EXIT

ruby server.rb 8000
