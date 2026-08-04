#!/usr/bin/env bash
# Stellt nach einem Kaltstart die Prüfumgebung wieder her.
#
# Die Sandbox ist flüchtig: wird der Container eingezogen, ist alles weg,
# was nicht im Repository liegt. Dieses Skript baut genau das wieder auf,
# was die Prüfungen aus .github/workflows/ brauchen — nicht mehr.
#
# Bewusst NICHT installiert: torch und pyannote.audio. Die Analyse läuft
# auf dem Hugging-Face-Space, nicht hier; lokal wären das mehrere Gigabyte
# ohne Nutzen.
set -uo pipefail

# Nur in der entfernten Umgebung — auf einem eigenen Rechner soll das
# Skript die vorhandene Installation nicht anfassen.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}" || exit 0

meldung() { echo "[Startaufbau] $*" >&2; }

# --- Git-Hooks scharfstellen -------------------------------------------
# Der Geheimnis-Prüfer greift nur, wenn Git ihn kennt. Nach einem frischen
# Klon ist die Einstellung weg, deshalb hier bei jedem Start.
if [ -d .githooks ]; then
  git config core.hooksPath .githooks
  meldung "Git-Hooks aktiv (.githooks)"
fi

# --- Python-Pakete -----------------------------------------------------
# Nur nachinstallieren, was wirklich fehlt: das Basis-Abbild bringt pytest,
# pyyaml und requests schon mit, und ein Lauf ohne Arbeit spart Sekunden
# bei jedem Sitzungsstart.
fehlend=()
pruefe_modul() {                     # $1 = Importname, $2 = Paketname
  python3 -c "import $1" 2>/dev/null || fehlend+=("$2")
}
pruefe_modul pytest pytest
pruefe_modul yaml pyyaml
pruefe_modul playwright playwright        # Oberflächentest im iPhone-Format
pruefe_modul gradio_client gradio_client   # Vollprüfung gegen das Backend

if [ ${#fehlend[@]} -gt 0 ]; then
  meldung "installiere: ${fehlend[*]}"
  python3 -m pip install --quiet --disable-pip-version-check \
    --root-user-action=ignore "${fehlend[@]}" \
    || meldung "WARNUNG: pip-Installation unvollständig"
else
  meldung "Python-Pakete vollständig"
fi

# --- Browser für den Oberflächentest -----------------------------------
# Chromium liegt im Abbild unter PLAYWRIGHT_BROWSERS_PATH. Nur wenn es
# dort wirklich fehlt, wird geladen — sonst zieht jeder Start 150 MB.
browser_pfad="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
if compgen -G "$browser_pfad/chromium*" >/dev/null; then
  meldung "Chromium vorhanden ($browser_pfad)"
else
  meldung "Chromium fehlt — lade nach"
  python3 -m playwright install chromium || meldung "WARNUNG: Chromium fehlt weiterhin"
fi

# --- Ruby-Übung --------------------------------------------------------
# Der ursprüngliche Lab-Teil des Repositories. Schlägt das fehl, ist das
# kein Grund, den Start zu blockieren.
gem_bin=""
if [ -d spec ] && command -v gem >/dev/null 2>&1; then
  # Das Verzeichnis, in dem Gems ihre Programme ablegen, liegt hier nicht
  # im PATH — ohne diesen Zusatz ist rspec trotz Installation nicht
  # aufrufbar.
  gem_bin="$(ruby -e 'require "rubygems"; puts Gem.bindir' 2>/dev/null)"
  [ -n "$gem_bin" ] && export PATH="$gem_bin:$PATH"

  if ! command -v rspec >/dev/null 2>&1; then
    meldung "installiere rspec"
    gem install --no-document --silent rspec >/dev/null 2>&1 \
      || meldung "Hinweis: rspec-Installation fehlgeschlagen"
  fi
  command -v rspec >/dev/null 2>&1 \
    && meldung "rspec bereit" \
    || meldung "Hinweis: rspec nicht aufrufbar"
fi

# --- Umgebung für die Sitzung merken -----------------------------------
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1" >> "$CLAUDE_ENV_FILE"
  [ -n "$gem_bin" ] && echo "export PATH=\"$gem_bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

meldung "bereit"
exit 0
