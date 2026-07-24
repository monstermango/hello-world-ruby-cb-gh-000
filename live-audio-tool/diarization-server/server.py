#!/usr/bin/env python3
"""Serverseitige Sprecher-Diarisierung für das Live-Audio-Analyse-Tool.

FastAPI-Service mit zwei ML-Engines:

* "embedded" (Standard): Resemblyzer-Sprecher-Embeddings + agglomeratives
  Clustering. Die Modellgewichte liegen im PyPI-Paket – kein Token, kein
  externer Download, läuft komplett offline.
* "pyannote" (optional): pyannote/speaker-diarization-3.1 – präziseres,
  zugangsbeschränktes Modell von Hugging Face. Wird automatisch benutzt,
  wenn HF_TOKEN gesetzt ist und das Modell geladen werden kann; sonst
  fällt der Server auf "embedded" zurück.

Start:
    uvicorn server:app --port 8001

Endpunkte:
    GET  /health   -> Status & verfügbare Engines
    POST /diarize  -> multipart "audio" (webm/ogg/wav/...), optional
                      "num_speakers"; Antwort:
                      {"segments": [{"start","end","speaker"}...],
                       "speakers": [...], "engine": "embedded"|"pyannote"}
"""

import os
import shutil
import subprocess
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PYANNOTE_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
MAX_SPEAKERS = 6

app = FastAPI(title="Diarization Server", version="2.0")

# Das Frontend läuft auf einem anderen Port (oder file://) – CORS freigeben.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pyannote_pipeline = None
_voice_encoder = None


def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# ----------------------------- Engine: pyannote -----------------------------

def get_pyannote():
    """Lädt die pyannote-Pipeline beim ersten Aufruf. Wirft bei Problemen."""
    global _pyannote_pipeline
    if _pyannote_pipeline is not None:
        return _pyannote_pipeline

    import torch
    from pyannote.audio import Pipeline

    try:
        # pyannote.audio >= 4.0
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=hf_token())
    except TypeError:
        # pyannote.audio 3.x
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token())
    if pipeline is None:
        raise RuntimeError(f"Pipeline {PYANNOTE_MODEL} konnte nicht geladen werden "
                           "(Modell-Lizenz auf Hugging Face akzeptiert?)")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    _pyannote_pipeline = pipeline
    return _pyannote_pipeline


def diarize_pyannote(wav_path, num_speakers):
    pipeline = get_pyannote()
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline(wav_path, **kwargs)
    return [
        {"start": round(turn.start, 2), "end": round(turn.end, 2), "speaker": label}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


# ----------------------------- Engine: embedded -----------------------------

def get_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder
        _voice_encoder = VoiceEncoder("cpu")
    return _voice_encoder

# Embedding-Fenster pro Sekunde; bestimmt die zeitliche Auflösung (~1/RATE s)
EMBED_RATE = 4
MIN_SEGMENT = 0.5          # kürzere Segmente werden verworfen (Sekunden)
MERGE_GAP = 0.4            # Lücken bis zu dieser Länge werden überbrückt
AUTO_DISTANCE = 0.40       # Cluster-Schwelle (Kosinusdistanz), wenn Anzahl unbekannt


def diarize_embedded(wav_path, num_speakers):
    import librosa
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    wav, sr = librosa.load(wav_path, sr=16000, mono=True)
    if len(wav) < sr:  # unter 1 Sekunde ist nichts zu holen
        return []

    encoder = get_encoder()
    _, partial_embeds, wav_splits = encoder.embed_utterance(
        wav, return_partials=True, rate=EMBED_RATE
    )
    embeds = np.asarray(partial_embeds)
    times = np.array([(s.start + s.stop) / 2 / sr for s in wav_splits])
    starts = np.array([s.start / sr for s in wav_splits])
    stops = np.array([s.stop / sr for s in wav_splits])

    # Energie-basierte Sprachaktivität: leise Fenster nicht clustern
    frame = int(sr / EMBED_RATE)
    rms_all = np.array([
        np.sqrt(np.mean(wav[max(0, int(t * sr) - frame // 2): int(t * sr) + frame // 2] ** 2) + 1e-12)
        for t in times
    ])
    threshold = max(0.008, float(np.percentile(rms_all, 25)) * 1.5)
    active = rms_all > threshold
    if active.sum() < 2:
        return []

    X = embeds[active]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    if num_speakers and num_speakers > 0:
        n = min(num_speakers, len(X))
        clustering = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=AUTO_DISTANCE,
            metric="cosine", linkage="average",
        )
    labels = clustering.fit_predict(X)

    # Zu viele Cluster? Kleinste zum nächstgelegenen großen schlagen.
    unique = list(np.unique(labels))
    if len(unique) > MAX_SPEAKERS:
        centroids = {u: X[labels == u].mean(axis=0) for u in unique}
        keep = sorted(unique, key=lambda u: -(labels == u).sum())[:MAX_SPEAKERS]
        for u in unique:
            if u in keep:
                continue
            best = max(keep, key=lambda k: float(np.dot(centroids[u], centroids[k])))
            labels[labels == u] = best

    # Label-Nummern nach erstem Auftreten vergeben (SPEAKER_00, SPEAKER_01, …)
    order = {}
    for lab in labels:
        if lab not in order:
            order[lab] = len(order)

    # Aktive Fenster mit Labels zu Segmenten zusammenfassen
    segments = []
    idx = np.where(active)[0]
    cur = None
    for j, i in enumerate(idx):
        lab = order[labels[j]]
        s, e = float(starts[i]), float(stops[i])
        if cur and cur["label"] == lab and s - cur["end"] <= MERGE_GAP:
            cur["end"] = e
        else:
            if cur:
                segments.append(cur)
            cur = {"label": lab, "start": s, "end": e}
    if cur:
        segments.append(cur)

    return [
        {"start": round(s["start"], 2), "end": round(s["end"], 2),
         "speaker": f"SPEAKER_{s['label']:02d}"}
        for s in segments
        if s["end"] - s["start"] >= MIN_SEGMENT
    ]


# ----------------------------- HTTP-Endpunkte -----------------------------

def embedded_available():
    try:
        import resemblyzer  # noqa: F401
        import sklearn      # noqa: F401
        return True
    except ImportError:
        return False


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
        "engines": {
            "embedded": embedded_available(),
            "pyannote": bool(hf_token()),
        },
        "pyannote_model": PYANNOTE_MODEL,
        "pyannote_loaded": _pyannote_pipeline is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.post("/diarize")
async def diarize(
    audio: UploadFile = File(...),
    num_speakers: int | None = Form(default=None),
    engine: str | None = Form(default=None),
):
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

        # Engine wählen: Wunsch des Clients > pyannote (falls Token) > embedded
        want = engine or ("pyannote" if hf_token() else "embedded")
        used = want
        if want == "pyannote":
            try:
                segments = diarize_pyannote(wav, num_speakers)
            except Exception as exc:
                if engine == "pyannote":  # explizit angefordert -> Fehler melden
                    raise HTTPException(status_code=503, detail=f"pyannote nicht verfügbar: {exc}")
                used = "embedded"  # automatischer Rückfall
                segments = run_embedded(wav, num_speakers)
        else:
            segments = run_embedded(wav, num_speakers)

    speakers = sorted({s["speaker"] for s in segments})
    return {"segments": segments, "speakers": speakers, "engine": used}


def run_embedded(wav, num_speakers):
    if not embedded_available():
        raise HTTPException(
            status_code=503,
            detail="Eingebaute Engine fehlt – bitte `pip install -r requirements.txt` ausführen.",
        )
    return diarize_embedded(wav, num_speakers)
