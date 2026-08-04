# Sprecher-Analyse — Aufbau, Betrieb, Datenschutz

## Überblick

Drei getrennte Bausteine, jeder mit einer klaren Aufgabe:

| Baustein | Ort | Aufgabe |
|---|---|---|
| **App (PWA)** | Static Space `Monstermango/sprecher-analyse` | Oberfläche, Aufnahme, Ablaufsteuerung |
| **API-Backend** | ZeroGPU-Space `Monstermango/diarization` | Sprechererkennung, Transkription, Alignment |
| **Datenablage** | Privates Dataset `Monstermango/diarization-results` | Transkripte als Markdown, Glossar |

Die Trennung ist Absicht: Die Oberfläche ist frei gestaltbar und kostet
keine Rechenzeit, während das Backend die Modelle hält. Beide sprechen
über eine schmale JSON-Schnittstelle miteinander.

## Verarbeitungskette

```
Audio  →  Vorbereiten  →  Sprecher erkennen  →  Transkribieren  →  Abschließen
          (WAV, Pegel)     (ganze Datei)        (Fenster für       (zusammen-
                                                 Fenster)           führen)
```

Jeder Schritt ist eine **eigene HTTP-Anfrage**. Das ist keine Zierde,
sondern notwendig: Die GPU-Zuteilung eines einzelnen Aufrufs läuft nach
kurzer Zeit ab, eine Stunde Audio passt niemals in eine Anfrage. Der
Server hält den Zwischenstand drei Stunden vor, sodass ein Abbruch
fortgesetzt werden kann.

Die **Sprechererkennung läuft immer über die gesamte Datei**, sonst wäre
„Sprecher 1" am Anfang nicht dieselbe Person wie am Ende. Nur die
Transkription wird gefenstert, mit Schnitten auf Sprecherwechseln.

Der Text wird **wortweise** zugeordnet: Jeder erkannte Abschnitt wird per
Forced Alignment in Einzelwörter zerlegt, jedes Wort bekommt den zu
diesem Zeitpunkt aktiven Sprecher. Satzweise Zuordnung verteilt kurze
Wortwechsel systematisch falsch.

## Zwei Erkenner zur Wahl

Für die Worterkennung stehen zwei Modelle bereit:

| Wahl | Modell | Bemerkung |
|---|---|---|
| `whisper` (Vorgabe) | `primeline/whisper-large-v3-german` | erprobt; nimmt Vorwissen nur nachträglich über das Glossar auf |
| `qwen` | `Qwen/Qwen3-ASR-1.7B` | nimmt Vorwissen unmittelbar als Kontext entgegen; liefert keine Zeitmarken |

Weil Qwen keine Zeitmarken liefert, läuft sein Text durch denselben
Aligner (`MMS_FA`) wie ein eingefügtes Fremdtranskript. Damit
unterscheidet sich zwischen zwei Läufen wirklich nur die Worterkennung —
Sprechertrennung, Ausrichtung und alles danach bleiben gleich.

Das ist der ganze Zweck der Auswahl: dieselbe Aufnahme zweimal laufen
lassen (Transkriptfeld dabei leer), beide Ergebnisse mit
`tests/wortfehlerrate.py` gegen das Transkript aus Sprachmemos messen und
danach begründet entscheiden. Welches Modell ein Protokoll erzeugt hat,
steht in dessen Frontmatter unter `modell`; ohne das ließe sich eine
gemessene Rate hinterher keinem Erkenner mehr zuordnen.

Die Gewichte werden **vor** der ersten GPU-Reservierung geholt. Innerhalb
wäre der Download aus dem reservierten Zeitfenster bezahlt, und der erste
Abschnitt liefe mittendrin auf.

## Aufteilung des Backends

| Datei | Inhalt | Testbar ohne GPU |
|---|---|---|
| `kern.py` | Fachlogik: Fensterung, Zuordnung, Markdown, Glossar, Pfadprüfung | ja |
| `app.py` | Modelle, GPU-Aufrufe, Sitzungen, API | nein |

`kern.py` kommt ohne Torch, Netz und GPU aus. Dadurch läuft die gesamte
Fachlogik in unter einer Sekunde durch die Tests — genau dort, wo in der
Entwicklung die meisten Fehler entstanden sind.

## Betrieb und Prüfung

- `tests/test_kern.py` — Fachlogik, 35 Prüfungen, ohne Hardware
- `tests/test_frontend.py` — kompletter Ablauf der App im iPhone-Format,
  Backend gemockt
- `tests/gesundheitscheck.py` — prüft Erreichbarkeit, startet den Space
  bei einem Fehlerzustand neu
- `.github/workflows/tests.yml` — beides bei jeder Änderung
- `.github/workflows/gesundheit.yml` — alle sechs Stunden, mit Neustart

Selbstheilung gibt es auf drei Ebenen: Die App wiederholt abgebrochene
Schritte, unterbrochene Analysen lassen sich fortsetzen, und ein Space im
Fehlerzustand wird automatisch neu gestartet. **Echte Programmfehler
behebt das nicht** — dagegen helfen die Tests vor dem Ausliefern.

## Datenschutz

Was wo liegt:

- **Aufnahmen** verlassen den Space nicht und werden **sofort nach der
  Analyse gelöscht** — Originalupload wie umgewandelte Fassung. Ein
  Aufräumlauf alle zehn Minuten fängt abgebrochene Sitzungen ab.
- **Transkripte und Glossar** liegen im privaten Dataset, erreichbar nur
  mit dem Token des Kontos.
- **Keine Drittanbieter**: Erkennung und Zuordnung laufen vollständig auf
  dem eigenen Space. Es gibt keine externe Transkriptions-API.

Zugriffsschutz:

- Jeder Endpunkt prüft den Zugangsschlüssel **vor** jeder Verarbeitung,
  laufzeitkonstant und mit wachsender Verzögerung nach Fehlversuchen.
- Nur Dateien aus `aufnahmen/` mit unverdächtigem Namen sind lesbar.
- Sitzungskennungen sind kryptografisch zufällig.

Bekannte Abwägungen:

- Der Backend-Space ist **öffentlich**, weil iOS die Anmeldung eines
  privaten Space über Fremd-Cookies blockiert. Geschützt wird deshalb
  über den Zugangsschlüssel, nicht über die Sichtbarkeit.
- Das hinterlegte HF-Token hat **Schreibrechte auf das ganze Konto**.
  Sicherer wäre ein fein abgestuftes Token, das nur das Ergebnis-Dataset
  beschreiben darf. Siehe unten.
- Das Glossar enthält **echte Namen**. Das ist gewollt, damit die
  Erkennung besser wird — es liegt im privaten Dataset.

## Empfohlene Nachbesserung

Ein fein abgestuftes Token anlegen
(huggingface.co/settings/tokens → *Fine-grained*) mit genau diesen Rechten:

- Lesen: die gated pyannote-Modelle
- Lesen und Schreiben: `Monstermango/diarization-results`

Dieses Token als Space-Geheimnis `HF_TOKEN` hinterlegen und das bisherige
Schreibtoken widerrufen. Damit kann ein kompromittierter Space nicht mehr
auf andere Repositories des Kontos zugreifen.
