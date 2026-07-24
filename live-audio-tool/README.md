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
