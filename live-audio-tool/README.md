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

## Starten

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
- Die Sprecherzuordnung ist **heuristisch** – sie trennt am zuverlässigsten Stimmen mit deutlich
  unterschiedlicher Tonlage (z. B. hohe vs. tiefe Stimme). Für produktionsreife Diarisierung
  wäre ein ML-Modell (z. B. pyannote.audio serverseitig) der nächste Ausbauschritt.
- Audio wird lokal im Browser verarbeitet; nur die Web-Speech-Transkription nutzt je nach
  Browser einen Dienst des Browserherstellers.
