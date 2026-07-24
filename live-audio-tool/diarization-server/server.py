#!/usr/bin/env python3
"""Serverseitige Sprecher-Diarisierung für das Live-Audio-Analyse-Tool.

FastAPI-Service, der eine Audioaufnahme entgegennimmt und mit pyannote.audio
(ML-Modell "pyannote/speaker-diarization-3.1") präzise Sprechersegmente
bestimmt. Das Frontend nutzt ihn, um seine heuristische Live-Zuordnung
nachträglich zu korrigieren.

Start:
    export HF_TOKEN=hf_...            # Hugging-Face-Token (Modell-Lizenz akzeptieren!)
    uvicorn server:app --port 8001

Endpunkte:
    GET  /health   -> Status & ob das Modell geladen ist
    POST /diarize  -> multipart "audio" (webm/ogg/wav/...), optional "num_speakers"
                      Antwort: {"segments": [{"start", "end", "speaker"}, ...]}

Das ML-Modell wird erst beim ersten /diarize-Aufruf geladen (Lazy Loading),
damit der Service auch ohne installierte ML-Abhängigkeiten startet und
/health beantworten kann.
"""

import os
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MODEL_NAME = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

app = FastAPI(title="Diarization Server", version="1.0")

# Das Frontend läuft auf einem anderen Port (oder file://) – CORS freigeben.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    """Lädt die pyannote-Pipeline beim ersten Aufruf (dauert einige Sekunden)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise HTTPException(
            status_code=503,
            detail="HF_TOKEN ist nicht gesetzt. Hugging-Face-Token mit akzeptierter "
                   f"Lizenz für {MODEL_NAME} wird benötigt.",
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"ML-Abhängigkeiten fehlen ({exc}). "
                   "Bitte `pip install -r requirements.txt` ausführen.",
        )

    pipeline = Pipeline.from_pretrained(MODEL_NAME, use_auth_token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    _pipeline = pipeline
    return _pipeline


def to_wav(src_path: str, dst_path: str) -> None:
    """Konvertiert beliebige Audioformate zu 16-kHz-Mono-WAV (per ffmpeg)."""
    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=503,
            detail="ffmpeg ist nicht installiert – wird für die Formatkonvertierung benötigt.",
        )
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        capture_output=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="Audiodatei konnte nicht dekodiert werden: "
                   + result.stderr.decode(errors="replace")[-400:],
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": _pipeline is not None,
        "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.post("/diarize")
async def diarize(
    audio: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
):
    pipeline = get_pipeline()

    suffix = os.path.splitext(audio.filename or "aufnahme.webm")[1] or ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "input" + suffix)
        wav = os.path.join(tmp, "input.wav")
        with open(src, "wb") as f:
            f.write(await audio.read())

        if suffix.lower() == ".wav":
            wav = src
        else:
            to_wav(src, wav)

        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        annotation = pipeline(wav, **kwargs)

    segments = [
        {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": label}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    speakers = sorted({s["speaker"] for s in segments})
    return {"segments": segments, "speakers": speakers}
