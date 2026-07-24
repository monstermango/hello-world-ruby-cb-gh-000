# Diarisierungs-Server (ML)

Serverseitige Sprecher-Diarisierung mit **pyannote.audio** – der Ausbauschritt für
präzise Sprecherzuordnung. Das Browser-Tool schneidet die Aufnahme mit und schickt
sie nach dem Stopp hierher; die zurückgelieferten Segmente korrigieren die
heuristische Live-Zuordnung (Transkript, Redeanteile).

## Voraussetzungen

1. **Python ≥ 3.10** und **ffmpeg** (für die Konvertierung von WebM/Opus zu WAV):
   ```sh
   sudo apt install ffmpeg        # bzw. brew install ffmpeg
   ```
2. **Hugging-Face-Token**: Das Modell `pyannote/speaker-diarization-3.1` ist
   zugangsbeschränkt („gated“). Einmalig:
   - Nutzungsbedingungen akzeptieren: <https://huggingface.co/pyannote/speaker-diarization-3.1>
     (und <https://huggingface.co/pyannote/segmentation-3.0>)
   - Token erstellen: <https://huggingface.co/settings/tokens>

## Installation & Start

```sh
cd live-audio-tool/diarization-server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # lädt u. a. PyTorch (mehrere hundert MB)

export HF_TOKEN=hf_...                   # dein Hugging-Face-Token
uvicorn server:app --port 8001
```

Beim ersten `/diarize`-Aufruf wird das Modell von Hugging Face geladen und gecacht.
Mit GPU (CUDA) läuft die Analyse deutlich schneller, CPU funktioniert aber auch.

## API

- `GET /health` – Status, u. a. ob Token gesetzt, ffmpeg vorhanden und Modell geladen ist.
- `POST /diarize` – Multipart-Upload:
  - `audio`: Audiodatei (webm, ogg, wav, mp3, …)
  - `num_speakers` (optional): Sprecheranzahl, falls bekannt – verbessert das Ergebnis.

  Antwort:
  ```json
  {
    "segments": [
      { "start": 0.5, "end": 4.2, "speaker": "SPEAKER_00" },
      { "start": 4.6, "end": 9.1, "speaker": "SPEAKER_01" }
    ],
    "speakers": ["SPEAKER_00", "SPEAKER_01"]
  }
  ```

## Zusammenspiel mit dem Frontend

Das Frontend prüft beim Laden `http://localhost:8001/health`. Ist der Server
erreichbar, erscheint nach dem Stoppen einer Aufnahme der Button
**„ML-Analyse“**: Die komplette Aufnahme wird hochgeladen, die ML-Segmente
ersetzen die heuristische Sprecherzuordnung, und Transkript sowie Redeanteile
werden neu berechnet. Ein anderer Server-Port/-Host lässt sich im Browser per
`localStorage.setItem("mlServer", "http://host:port")` hinterlegen.
