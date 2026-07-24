# Diarisierungs-Server (ML)

Serverseitige Sprecher-Diarisierung für das Live-Audio-Tool – mit **zwei Engines**:

| Engine | Modell | Voraussetzungen | Qualität |
|---|---|---|---|
| **embedded** (Standard) | Resemblyzer-Sprecher-Embeddings + Clustering | keine – läuft komplett offline | gut |
| **pyannote** (optional) | pyannote/speaker-diarization-3.1 | Hugging-Face-Token + Modell-Lizenz | sehr gut |

Die eingebaute Engine funktioniert **ohne Token und ohne externe Downloads** –
die Modellgewichte stecken im PyPI-Paket. pyannote wird automatisch benutzt,
sobald `HF_TOKEN` gesetzt ist; schlägt das fehl, fällt der Server auf die
eingebaute Engine zurück.

## Installation & Start

Am einfachsten über das Startskript im Tool-Verzeichnis (startet Web-App und
ML-Server zusammen):

```sh
./live-audio-tool/run.sh
```

Oder manuell:

```sh
cd live-audio-tool/diarization-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # lädt u. a. PyTorch (mehrere hundert MB)
uvicorn server:app --port 8001
```

**ffmpeg** wird für die Konvertierung der Browser-Aufnahmen benötigt:
`sudo apt install ffmpeg` (bzw. `brew install ffmpeg`).

### Optional: pyannote-Engine aktivieren

1. Nutzungsbedingungen akzeptieren:
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   und [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
2. Token erstellen: <https://huggingface.co/settings/tokens> (Read genügt)
3. ```sh
   pip install -r requirements-pyannote.txt
   export HF_TOKEN=hf_...
   uvicorn server:app --port 8001
   ```

## API

- `GET /health` – Status: verfügbare Engines, ffmpeg, geladenes Modell.
- `POST /diarize` – Multipart-Upload:
  - `audio`: Audiodatei (webm, ogg, wav, mp3, …)
  - `num_speakers` (optional): Sprecheranzahl, falls bekannt – verbessert das Ergebnis.
  - `engine` (optional): `embedded` oder `pyannote` erzwingen.

  Antwort:
  ```json
  {
    "segments": [
      { "start": 0.5, "end": 4.85, "speaker": "SPEAKER_00" },
      { "start": 4.25, "end": 9.6, "speaker": "SPEAKER_01" }
    ],
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "engine": "embedded"
  }
  ```

## Zusammenspiel mit dem Frontend

Das Frontend prüft beim Laden `http://localhost:8001/health`. Ist der Server
erreichbar, erscheint nach dem Stoppen einer Aufnahme der Button
**„ML-Analyse“**: Die Aufnahme wird hochgeladen, die ML-Segmente ersetzen die
heuristische Sprecherzuordnung, und Transkript sowie Redeanteile werden neu
berechnet. Ein anderer Server lässt sich im Browser per
`localStorage.setItem("mlServer", "http://host:port")` hinterlegen.

## Funktionsweise der eingebauten Engine

1. Audio → 16-kHz-Mono-WAV (ffmpeg)
2. Sprecher-Embeddings über gleitende Fenster (Resemblyzer, 4 Fenster/s)
3. Energie-basierte Sprachaktivitätserkennung (leise Fenster werden ignoriert)
4. Agglomeratives Clustering der Embeddings (Kosinusdistanz); Sprecheranzahl
   automatisch über eine Distanzschwelle oder per `num_speakers` vorgegeben
5. Fenster gleicher Sprecher werden zu Segmenten zusammengefasst
