# Absicherung der Entwicklungsumgebung

Drei Schutzschichten, damit typische Fehler auffallen, bevor sie Schaden
anrichten — plus ein Startaufbau, der die Umgebung nach einem Kaltstart
wiederherstellt.

## 1. Zugangsdaten kommen nicht ins Repository

`geheimnisse_pruefen.sh` durchsucht alles, was gleich committet würde,
nach Mustern echter Zugangsdaten (Hugging Face, OpenAI, GitHub, AWS,
Slack, private Schlüssel). Es prüft nur **hinzugefügte** Zeilen — gelöschte
sind unbedenklich.

Der Prüfer läuft an zwei Stellen:

- **Git-Hook** `.githooks/pre-commit` — greift bei jedem Commit, egal von
  wem. Einmalig aktivieren:

  ```
  git config core.hooksPath .githooks
  ```

- **Claude-Hook** in `.claude/settings.json` — greift schon *bevor* Claude
  `git commit` oder `git push` ausführt, und meldet den Fund zurück,
  sodass er sofort behoben wird.

GitHubs eigene Push-Protection bleibt als dritte, unabhängige Instanz
bestehen. Genau die hat in dieser Entwicklung ein echtes Leck abgefangen —
deshalb sind die zwei früheren Schichten dazugekommen.

## 2. Kerntests laufen vor dem Abschluss

`kerntests.sh` läuft als Stop-Hook, wenn Claude fertig zu sein glaubt.
Schlagen die Tests fehl, wird das zurückgemeldet statt stillschweigend
übergangen. Wurde nichts am Projekt geändert, beendet sich der Hook
sofort — unbeteiligte Sitzungen merken nichts davon.

## 3. Grenzen für Werkzeuge

`.claude/settings.json` verbietet zerstörerische Befehle (`rm -rf /`,
Pipe-nach-Shell aus dem Netz) und verlangt Rückfrage bei
`git push --force` und `git reset --hard`. Lesende Befehle laufen ohne
Nachfrage durch.

## 4. Umgebung übersteht den Kaltstart

Die Sandbox von Claude Code im Web ist flüchtig: Nach einer Weile ohne
Aktivität wird der Container eingezogen, beim nächsten Zugriff startet ein
frischer. Alles, was nicht im Repository liegt, ist dann weg — auch
nachinstallierte Pakete.

`.claude/hooks/session-start.sh` läuft als SessionStart-Hook und baut das
wieder auf, was die Prüfungen aus `.github/workflows/` brauchen:

- fehlende Python-Pakete (`playwright`, `gradio_client`; `pytest`, `pyyaml`
  bringt das Abbild mit)
- `rspec` samt PATH-Eintrag, falls `spec/` vorhanden ist
- `git config core.hooksPath .githooks` — sonst greift Schicht 1 nach einem
  frischen Klon nicht

Bewusst **nicht** installiert werden `torch` und `pyannote.audio`. Die
Analyse läuft auf dem Hugging-Face-Space; lokal wären das mehrere Gigabyte
ohne Nutzen.

Das Skript ist idempotent und prüft vor jeder Installation, ob sie nötig
ist — der zweite Lauf braucht unter einer Sekunde. Chromium liegt bereits
unter `PLAYWRIGHT_BROWSERS_PATH` und wird nur nachgeladen, wenn es dort
wirklich fehlt. Außerhalb der entfernten Umgebung beendet sich der Hook
sofort und ohne Ausgabe, damit er lokale Installationen nicht anfasst.

## Prüfen, ob alles greift

```
./scripts/geheimnisse_pruefen.sh            # 0 = sauber
./scripts/kerntests.sh                      # 0 = grün, 2 = Fehler
git config core.hooksPath                   # muss .githooks ergeben
CLAUDE_CODE_REMOTE=true CLAUDE_PROJECT_DIR="$PWD" \
  ./.claude/hooks/session-start.sh          # 0 = Umgebung steht
```
