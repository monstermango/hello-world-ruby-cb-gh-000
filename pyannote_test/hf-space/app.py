"""API-Backend der Sprecher-Analyse (HF Space auf ZeroGPU).

Die eigentliche App (PWA) läuft als eigener Static Space und spricht die
hier registrierten API-Endpunkte an. Jeder Endpunkt verlangt den
Zugangsschlüssel (Secret APP_PASS); erst nach der Prüfung wird GPU-Zeit
verbraucht.
"""
import os
import subprocess
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import spaces
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from transformers import pipeline as hf_pipeline

HF_TOKEN = os.environ["HF_TOKEN"]
APP_KEY = os.environ["APP_PASS"]
DATEN_REPO = "Monstermango/diarization-results"
FRONTEND = "https://monstermango-sprecher-analyse.static.hf.space"
ZEITZONE = ZoneInfo("Europe/Berlin")

api = HfApi(token=HF_TOKEN)
api.create_repo(DATEN_REPO, repo_type="dataset", private=True, exist_ok=True)

diarizer = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=HF_TOKEN
)
diarizer.to(torch.device("cuda"))

asr = hf_pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device="cuda",
)

loader = Audio(sample_rate=16000, mono="downmix")

FARBEN = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
          "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]


def _zeit(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d}"


def _namen_farben(labels):
    namen = {lb: f"Sprecher {i + 1}" for i, lb in enumerate(sorted(labels))}
    farben = {lb: FARBEN[i % len(FARBEN)] for i, lb in enumerate(sorted(labels))}
    return namen, farben


def _pruefe(key):
    return bool(key) and key == APP_KEY


@spaces.GPU(duration=120)
def _analyse_gpu(audio_path, num_speakers):
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = diarizer(audio_path, **kwargs)
    dia = getattr(output, "speaker_diarization", output)

    # Die ganze Aufnahme in einem Stück transkribieren: voller Kontext liefert
    # deutlich bessere Texte als Einzelsegmente; die Sprache erkennt Whisper
    # dabei automatisch.
    transkript = asr(audio_path, return_timestamps=True, return_language=True)
    chunks = []
    for c in transkript.get("chunks", []):
        start, ende = c.get("timestamp") or (None, None)
        if start is None or not c.get("text", "").strip():
            continue
        chunks.append({"start": float(start),
                       "ende": float(ende if ende is not None else start + 30.0),
                       "text": c["text"].strip(),
                       "sprache": c.get("language")})
    sprache = next((c["sprache"] for c in chunks if c.get("sprache")), None)

    turns = [(turn, lb) for turn, _, lb in dia.itertracks(yield_label=True)]
    texte = [[] for _ in turns]
    for c in chunks:
        mitte = (c["start"] + c["ende"]) / 2
        best, best_wert = None, None
        for i, (turn, _) in enumerate(turns):
            ueberlappung = max(0.0, min(c["ende"], turn.end)
                               - max(c["start"], turn.start))
            abstand = abs(mitte - (turn.start + turn.end) / 2)
            wert = (-ueberlappung, abstand)
            if best_wert is None or wert < best_wert:
                best, best_wert = i, wert
        if best is not None:
            texte[best].append(c["text"])

    segmente = []
    for i, (turn, lb) in enumerate(turns):
        segmente.append({"start": round(turn.start, 2),
                         "ende": round(turn.end, 2), "label": lb,
                         "text": " ".join(texte[i]) or None})

    stats = {lb: {"dauer": round(dia.label_duration(lb), 1),
                  "turns": len(dia.label_timeline(lb))}
             for lb in sorted(dia.labels())}

    overlaps = []
    for seg in dia.get_overlap():
        wer = sorted(lb for lb in dia.labels() if dia.label_timeline(lb).crop(seg))
        overlaps.append({"start": round(seg.start, 2),
                         "ende": round(seg.end, 2), "wer": wer})

    return {"dauer": round(loader.get_duration(audio_path), 1),
            "segmente": segmente, "stats": stats, "overlaps": overlaps,
            "sprache": sprache}


# ---------- Markdown-Ablage ----------

def _markdown(daten, quelle, zeitpunkt, namen=None):
    std_namen, _ = _namen_farben(daten["stats"])
    namen = {**std_namen, **(namen or {})}
    frontmatter = {
        "titel": f"Aufnahme {zeitpunkt:%Y-%m-%d %H:%M}",
        "datum": zeitpunkt.isoformat(timespec="seconds"),
        "dauer_s": daten["dauer"],
        "sprecher": len(daten["stats"]),
        "sprache": daten.get("sprache"),
        "quelle": quelle,
        "redeanteile_s": {namen[lb]: st["dauer"]
                          for lb, st in daten["stats"].items()},
    }
    md = ["---", yaml.safe_dump(frontmatter, allow_unicode=True,
                                sort_keys=False).strip(), "---", "",
          f"# {frontmatter['titel']}", ""]

    gesamt = sum(st["dauer"] for st in daten["stats"].values()) or 1.0
    md += ["## Redeanteile", "",
           "| Sprecher | Sprechzeit | Anteil | Redebeiträge |",
           "|---|---|---|---|"]
    for lb, st in sorted(daten["stats"].items(),
                         key=lambda x: x[1]["dauer"], reverse=True):
        md.append(f"| {namen[lb]} | {_zeit(st['dauer'])} min "
                  f"| {100 * st['dauer'] / gesamt:.0f} % | {st['turns']} |")

    md += ["", "## Protokoll", ""]
    for s in daten["segmente"]:
        if s["text"]:
            md.append(f"**{namen[s['label']]}** "
                      f"({_zeit(s['start'])}–{_zeit(s['ende'])}): {s['text']}")
            md.append("")

    md += ["## Segmente", "", "| Start | Ende | Sprecher |", "|---|---|---|"]
    for s in daten["segmente"]:
        md.append(f"| {_zeit(s['start'])} | {_zeit(s['ende'])} | {namen[s['label']]} |")

    if daten["overlaps"]:
        md += ["", "## Gleichzeitiges Sprechen", "",
               "| Start | Ende | Beteiligte |", "|---|---|---|"]
        for o in daten["overlaps"]:
            wer = ", ".join(namen[lb] for lb in o["wer"])
            md.append(f"| {_zeit(o['start'])} | {_zeit(o['ende'])} | {wer} |")

    return "\n".join(md) + "\n"


def _speichern(md_text, zeitpunkt):
    pfad = f"aufnahmen/{zeitpunkt:%Y-%m-%d_%H%M%S}.md"
    api.upload_file(path_or_fileobj=md_text.encode("utf-8"), path_in_repo=pfad,
                    repo_id=DATEN_REPO, repo_type="dataset",
                    commit_message=f"Analyse {pfad}")
    return pfad


def _frontmatter(text):
    if text.startswith("---"):
        try:
            ende = text.index("\n---", 3)
            return (yaml.safe_load(text[3:ende]) or {}), text[ende + 4:]
        except (ValueError, yaml.YAMLError):
            pass
    return {}, text


# ---------- API-Endpunkte ----------

def _nach_wav(pfad):
    """Konvertiert beliebige Audioformate (m4a, webm, …) nach 16-kHz-Mono-WAV."""
    ziel = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pfad,
                    "-ar", "16000", "-ac", "1", ziel], check=True)
    return ziel


def analysieren_api(key, audio, num_speakers):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    if not audio:
        return {"ok": False, "fehler": "Kein Audio übermittelt."}
    quelle = os.path.basename(audio)
    try:
        audio = _nach_wav(audio)
    except subprocess.CalledProcessError:
        return {"ok": False, "fehler": "Audioformat konnte nicht gelesen werden."}
    try:
        daten = _analyse_gpu(audio, num_speakers)
    except Exception as e:
        return {"ok": False, "fehler": f"Analyse fehlgeschlagen: {e}"}
    zeitpunkt = datetime.now(ZEITZONE)
    namen, farben = _namen_farben(daten["stats"])
    return {"ok": True, "daten": daten, "namen": namen, "farben": farben,
            "zeitpunkt": zeitpunkt.isoformat(timespec="seconds"),
            "quelle": quelle, "sprache": daten.get("sprache")}


def speichern_api(key, analyse, namen):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    try:
        daten = analyse["daten"]
        zeitpunkt = datetime.fromisoformat(analyse["zeitpunkt"])
        quelle = analyse.get("quelle") or "aufnahme"
        daten["stats"]
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "fehler": "Ungültige Analysedaten."}
    eigene = {k: v.strip() for k, v in (namen or {}).items()
              if isinstance(v, str) and v.strip()}
    md_text = _markdown(daten, quelle, zeitpunkt, eigene)
    try:
        pfad = _speichern(md_text, zeitpunkt)
    except Exception as e:
        return {"ok": False, "fehler": f"Speichern fehlgeschlagen: {e}"}
    return {"ok": True, "pfad": pfad, "markdown": md_text}


def verlauf_api(key):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    eintraege = []
    dateien = sorted(
        (f for f in api.list_repo_files(DATEN_REPO, repo_type="dataset")
         if f.startswith("aufnahmen/") and f.endswith(".md")), reverse=True)[:20]
    for f in dateien:
        lokal = hf_hub_download(DATEN_REPO, f, repo_type="dataset", token=HF_TOKEN)
        with open(lokal, encoding="utf-8") as fh:
            fm, _ = _frontmatter(fh.read())
        eintraege.append({"pfad": f,
                          "titel": fm.get("titel", f),
                          "datum": str(fm.get("datum", "")),
                          "sprecher": fm.get("sprecher"),
                          "dauer_s": fm.get("dauer_s")})
    return {"ok": True, "eintraege": eintraege}


def eintrag_api(key, pfad):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    if not (isinstance(pfad, str) and pfad.startswith("aufnahmen/")
            and pfad.endswith(".md") and ".." not in pfad):
        return {"ok": False, "fehler": "Ungültiger Pfad."}
    lokal = hf_hub_download(DATEN_REPO, pfad, repo_type="dataset", token=HF_TOKEN)
    with open(lokal, encoding="utf-8") as fh:
        text = fh.read()
    fm, body = _frontmatter(text)
    return {"ok": True, "frontmatter": fm, "body": body, "markdown": text}


with gr.Blocks(title="Sprecher-Analyse API") as demo:
    gr.Markdown(f"# 🎙️ Sprecher-Analyse — API-Backend\n"
                f"Die App selbst läuft unter **[{FRONTEND}]({FRONTEND})**. "
                f"Dieses Formular dient nur zu Testzwecken.")
    key = gr.Textbox(label="Zugangsschlüssel", type="password")
    with gr.Tab("Analyse"):
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                         label="Audio")
        num_speakers = gr.Number(value=0, precision=0,
                                 label="Anzahl Sprecher (0 = automatisch)")
        b_analyse = gr.Button("Analysieren", variant="primary")
        out_analyse = gr.JSON()
        b_analyse.click(analysieren_api, [key, audio, num_speakers],
                        out_analyse, api_name="analysieren")
        analyse_json = gr.JSON(label="Analyse-Objekt (aus /analysieren)")
        namen_json = gr.JSON(label='Sprechernamen, z. B. {"SPEAKER_00": "Nils"}')
        b_speichern = gr.Button("Als Markdown speichern")
        out_speichern = gr.JSON()
        b_speichern.click(speichern_api, [key, analyse_json, namen_json],
                          out_speichern, api_name="speichern")
    with gr.Tab("Verlauf"):
        b_verlauf = gr.Button("Verlauf laden")
        out_verlauf = gr.JSON()
        b_verlauf.click(verlauf_api, [key], out_verlauf, api_name="verlauf")
        pfad = gr.Textbox(label="Pfad (aufnahmen/....md)")
        b_eintrag = gr.Button("Eintrag laden")
        out_eintrag = gr.JSON()
        b_eintrag.click(eintrag_api, [key, pfad], out_eintrag, api_name="eintrag")

demo.launch(ssr_mode=False)
