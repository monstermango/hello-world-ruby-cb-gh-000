# Live-Audio-Analyse – Sprecher & Inhalte

Ein Browser-Tool, das **live Mikrofon-Audio** verarbeitet, Äußerungen **Sprechern zuordnet**
und die **Inhalte visualisiert** – komplett ohne externe Bibliotheken oder Build-Schritt.

## Features

- **Live-Visualisierung**: Wellenform und Frequenzspektrum in Echtzeit (Web Audio API),
  eingefärbt in der Farbe des gerade aktiven Sprechers.
- **Sprecherzuordnung (Diarisierung, heuristisch)**: Sprecher werden automatisch anhand von
  Stimmmerkmalen erkannt – Tonhöhe (Autokorrelation) und spektraler Schwerpunkt (Klangfarbe) –
  und per Online-Clustering mit Hysterese zugeordnet. Sprecher lassen sich umbenennen und
  bei Bedarf manuell „fixieren“.
- **Live-Transkription**: Über die Web Speech API (Chrome/Edge). Jedes finale Segment wird dem
  im Zeitraum dominanten Sprecher zugewiesen.
- **Inhalts-Visualisierung**:
  - Transkript-Feed mit Sprecherfarben und Zeitstempeln
  - Redeanteile pro Sprecher (Balken + Sprechzeit)
  - Häufigste Begriffe (Stopwörter DE/EN gefiltert, Größe nach Häufigkeit)
- **Export**: Transkript als JSON (Segmente, Sprecher, Keywords) oder TXT.
- **ML-Nachanalyse (optional)**: Läuft der [Diarisierungs-Server](diarization-server/),
  erscheint nach dem Stoppen der Button **„ML-Analyse“** – die Aufnahme wird
  serverseitig per Sprecher-Embeddings präzise diarisiert und die heuristische
  Zuordnung (Transkript, Redeanteile) automatisch korrigiert. Funktioniert ohne
  Token/Downloads; optional mit pyannote.audio für noch bessere Qualität.

## Starten

Alles zusammen (Web-App + ML-Server, richtet beim ersten Mal die Python-Umgebung ein):

```sh
./live-audio-tool/run.sh              # App: http://localhost:8000, ML: Port 8001
```

Nur die Web-App (ohne ML-Nachanalyse):

```sh
ruby live-audio-tool/server.rb        # startet auf http://localhost:8000
ruby live-audio-tool/server.rb 9090   # alternativer Port
```

Dann im Browser `http://localhost:8000` öffnen und **Aufnahme starten**.
Alternativ genügt es auch, `index.html` direkt in Chrome zu öffnen.

## Dauerbetrieb in der Cloud

Für den permanenten Betrieb auf einem eigenen Server (jederzeit vom Handy
erreichbar, mit eigener Domain, HTTPS und Passwortschutz) liegt in
[`deploy/`](deploy/) ein komplettes Docker-Compose-Setup samt Anleitung –
Kurzfassung: VPS mieten, DNS-Eintrag setzen, `.env` ausfüllen,
`docker compose up -d --build`.

## Vom Handy nutzen

Das Handy ist nur der Bildschirm + das Mikrofon – die Rechenarbeit läuft auf einem
Rechner (PC/Mac/Server), auf dem `./live-audio-tool/run.sh` gestartet wird. Der
Ruby-Server leitet `/ml/*` an den ML-Server weiter, daher reicht **eine** Adresse.

Der Mikrofonzugriff im Handy-Browser erfordert **HTTPS** – eine `http://192.168…`-
Adresse genügt nicht. Drei einfache Wege zu einer HTTPS-Adresse:

| Weg | Befehl auf dem Rechner | Bemerkung |
|---|---|---|
| **Tailscale** (empfohlen) | `tailscale serve 8000` | privat, kostenlos; Handy braucht die Tailscale-App im selben Konto |
| **cloudflared** | `cloudflared tunnel --url http://localhost:8000` | öffentliche Wegwerf-URL, kein Konto nötig |
| **ngrok** | `ngrok http 8000` | öffentliche URL, kostenloses Konto |

Die angezeigte `https://…`-Adresse auf dem Handy öffnen – fertig. Für **maximale
Qualität** vor dem Start auf dem Rechner `export HF_TOKEN=hf_…` setzen (siehe
[diarization-server/README.md](diarization-server/README.md)); `run.sh`
installiert die pyannote-Engine dann automatisch.

Auf iPhones läuft die Live-Transkription über Safari (Web Speech API, iOS 14.5+);
falls ein Browser sie nicht unterstützt, funktionieren Visualisierung,
Sprechererkennung und ML-Analyse trotzdem.

## Hinweise & Grenzen

- Mikrofonzugriff braucht einen *secure context* (localhost oder HTTPS).
- Die Transkription (Web Speech API) gibt es aktuell nur in Chromium-Browsern; in anderen
  Browsern funktionieren Visualisierung und Sprechererkennung trotzdem.
- Die **Live**-Sprecherzuordnung ist heuristisch – sie trennt am zuverlässigsten Stimmen mit
  deutlich unterschiedlicher Tonlage. Für präzise Ergebnisse gibt es die serverseitige
  ML-Nachanalyse mit pyannote.audio – Setup siehe
  [diarization-server/README.md](diarization-server/README.md).
- Audio wird lokal im Browser verarbeitet; nur die Web-Speech-Transkription nutzt je nach
  Browser einen Dienst des Browserherstellers.
