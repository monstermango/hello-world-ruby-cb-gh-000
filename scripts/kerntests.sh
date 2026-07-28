#!/usr/bin/env bash
# Führt die schnellen Kerntests aus — aber nur, wenn am Projekt gearbeitet
# wurde. So bleibt der Hook in unbeteiligten Sitzungen lautlos.
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

if ! git status --porcelain -- pyannote_test | grep -q .; then
  exit 0                      # nichts am Projekt geändert
fi

ausgabe="$(python3 -m pytest pyannote_test/tests/test_kern.py -q 2>&1)"
if [ $? -ne 0 ]; then
  {
    echo "Die Kerntests schlagen fehl — bitte beheben, bevor du fertig bist:"
    echo "$ausgabe" | tail -20
  } >&2
  exit 2                      # blockiert und meldet es Claude zurück
fi
exit 0
