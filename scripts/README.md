# Absicherung der Entwicklungsumgebung

Drei Schutzschichten, damit typische Fehler auffallen, bevor sie Schaden
anrichten.

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

## Prüfen, ob alles greift

```
./scripts/geheimnisse_pruefen.sh            # 0 = sauber
./scripts/kerntests.sh                      # 0 = grün, 2 = Fehler
git config core.hooksPath                   # muss .githooks ergeben
```
