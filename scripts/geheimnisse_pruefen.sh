#!/usr/bin/env bash
# Sucht Zugangsdaten in dem, was gleich committet würde.
#
# Aufruf ohne Argumente prüft den Staging-Bereich und die geänderten
# Dateien im Arbeitsverzeichnis — Letzteres, weil `git commit -a` erst
# beim Ausführen stagt.
#
# Rückgabe 0 = sauber, 1 = Fund (Meldung auf stderr).
set -uo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0

# Hohe Treffsicherheit statt Vollständigkeit: Muster, die praktisch nie
# zufällig in Quelltext stehen.
MUSTER='(hf_[A-Za-z0-9]{30,}
|sk-[A-Za-z0-9_-]{24,}
|gh[pousr]_[A-Za-z0-9]{30,}
|AKIA[0-9A-Z]{16}
|xox[baprs]-[A-Za-z0-9-]{10,}
|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
MUSTER="${MUSTER//$'\n'/}"

inhalt="$(git diff --cached --no-color; git diff --no-color)"
[ -z "$inhalt" ] && exit 0

# Nur hinzugefügte Zeilen prüfen; gelöschte Zeilen sind unbedenklich.
funde="$(printf '%s' "$inhalt" | grep -E '^\+' | grep -Eo "$MUSTER" | sort -u)"

if [ -n "$funde" ]; then
  {
    echo "Zugangsdaten im Commit gefunden — Push wurde verhindert:"
    printf '%s\n' "$funde" | while read -r fund; do
      echo "  ${fund:0:12}… ($(printf '%s' "$inhalt" | grep -c -- "$fund") Fundstellen)"
    done
    echo
    echo "Zugangsdaten gehören in Umgebungsvariablen, nicht ins Repository."
    echo "Beispiel: token = os.environ[\"HF_TOKEN\"]"
  } >&2
  exit 1
fi
exit 0
