#!/bin/bash
# SessionStart-Hook für Claude Code on the web:
# richtet alles ein, damit das Live-Audio-Tool inkl. ML-Server sofort
# lauffähig ist (ffmpeg, rspec, Python-Umgebung mit beiden ML-Engines).
set -euo pipefail

# Nur in Remote-Sessions (Claude Code on the web) ausführen
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# ffmpeg: Formatkonvertierung für den Diarisierungs-Server
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq || true
  apt-get install -y -qq ffmpeg
fi

# rspec: Ruby-Tests des Repos ("gem list -i" statt PATH-Check, da rbenv
# neue Binärdateien erst nach einem rehash in den PATH legt)
if ! gem list -i rspec >/dev/null 2>&1; then
  gem install rspec --no-document
  command -v rbenv >/dev/null 2>&1 && rbenv rehash
fi

# Python-Umgebung des ML-Servers (eingebaute Engine + pyannote)
ML_DIR="live-audio-tool/diarization-server"
if [ ! -x "$ML_DIR/.venv/bin/python" ]; then
  python3 -m venv "$ML_DIR/.venv"
fi
"$ML_DIR/.venv/bin/pip" install --quiet \
  -r "$ML_DIR/requirements.txt" \
  -r "$ML_DIR/requirements-pyannote.txt"

# rbenv-Shims (u. a. rspec) für die Session in den PATH legen
if [ -n "${CLAUDE_ENV_FILE:-}" ] && [ -d /opt/rbenv/shims ]; then
  echo 'export PATH="/opt/rbenv/shims:$PATH"' >> "$CLAUDE_ENV_FILE"
fi

echo "Session-Setup abgeschlossen: ffmpeg, rspec und ML-Umgebung sind bereit."
