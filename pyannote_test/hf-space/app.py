"""Sprecher-Analyse mit Dashboard und Markdown-Ablage (HF Space auf ZeroGPU)."""
import os
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
DATEN_REPO = "Monstermango/diarization-results"
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

USE_CASES = [
    "Wer spricht wann?",
    "Gesprächsprotokoll (mit Text)",
    "Redeanteile & Statistik",
    "Durcheinanderreden finden",
]


def _zeit(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d}"


def _namen_farben(labels):
    namen = {lb: f"Sprecher {i + 1}" for i, lb in enumerate(sorted(labels))}
    farben = {lb: FARBEN[i % len(FARBEN)] for i, lb in enumerate(sorted(labels))}
    return namen, farben


@spaces.GPU(duration=120)
def _analyse_gpu(audio_path, mit_text, num_speakers):
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = diarizer(audio_path, **kwargs)
    dia = getattr(output, "speaker_diarization", output)

    segmente = []
    for turn, _, lb in dia.itertracks(yield_label=True):
        text = None
        if mit_text and turn.duration >= 0.3:
            wellenform, sr = loader.crop(audio_path, Segment(turn.start, turn.end))
            text = asr({"array": wellenform.squeeze(0).numpy(),
                        "sampling_rate": sr})["text"].strip()
        segmente.append({"start": round(turn.start, 2),
                         "ende": round(turn.end, 2), "label": lb, "text": text})

    stats = {lb: {"dauer": round(dia.label_duration(lb), 1),
                  "turns": len(dia.label_timeline(lb))}
             for lb in sorted(dia.labels())}

    overlaps = []
    for seg in dia.get_overlap():
        wer = sorted(lb for lb in dia.labels() if dia.label_timeline(lb).crop(seg))
        overlaps.append({"start": round(seg.start, 2),
                         "ende": round(seg.end, 2), "wer": wer})

    return {"dauer": round(loader.get_duration(audio_path), 1),
            "segmente": segmente, "stats": stats, "overlaps": overlaps}


# ---------- Markdown-Ablage ----------

def _markdown(daten, use_case, quelle, zeitpunkt):
    namen, _ = _namen_farben(daten["stats"])
    frontmatter = {
        "titel": f"Aufnahme {zeitpunkt:%Y-%m-%d %H:%M}",
        "datum": zeitpunkt.isoformat(timespec="seconds"),
        "dauer_s": daten["dauer"],
        "anwendungsfall": use_case,
        "sprecher": len(daten["stats"]),
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

    md += ["", "## Segmente", "", "| Start | Ende | Sprecher |", "|---|---|---|"]
    for s in daten["segmente"]:
        md.append(f"| {_zeit(s['start'])} | {_zeit(s['ende'])} | {namen[s['label']]} |")

    if any(s["text"] for s in daten["segmente"]):
        md += ["", "## Protokoll", ""]
        for s in daten["segmente"]:
            if s["text"]:
                md.append(f"**{namen[s['label']]}** "
                          f"({_zeit(s['start'])}–{_zeit(s['ende'])}): {s['text']}")
                md.append("")

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


# ---------- HTML-Darstellung ----------

def _chip(text, farbe):
    return (f'<span style="background:{farbe};color:#fff;border-radius:12px;'
            f'padding:2px 10px;font-weight:600;white-space:nowrap">{text}</span>')


def _zeitleiste(daten, namen, farben):
    dauer = daten["dauer"] or 1.0
    lanes = []
    for lb in sorted(namen):
        bloecke = ""
        for s in daten["segmente"]:
            if s["label"] != lb:
                continue
            links = 100 * s["start"] / dauer
            breite = max(0.5, 100 * (s["ende"] - s["start"]) / dauer)
            bloecke += (f'<div style="position:absolute;left:{links:.2f}%;'
                        f'width:{breite:.2f}%;top:0;bottom:0;'
                        f'background:{farben[lb]};border-radius:3px"></div>')
        lanes.append(
            f'<div style="position:relative;height:18px;margin:4px 0;'
            f'background:rgba(128,128,128,.15);border-radius:3px">{bloecke}</div>')
    return ('<div style="margin:8px 0 16px 0">' + "".join(lanes) +
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:.8em;opacity:.7"><span>0:00</span>'
            f'<span>{_zeit(dauer)}</span></div></div>')


def _html(daten, use_case):
    namen, farben = _namen_farben(daten["stats"])
    if use_case == USE_CASES[1]:
        bloecke = ""
        for s in daten["segmente"]:
            if not s["text"]:
                continue
            bloecke += (f'<div style="margin:10px 0;padding-left:12px;'
                        f'border-left:4px solid {farben[s["label"]]}">'
                        f'{_chip(namen[s["label"]], farben[s["label"]])} '
                        f'<span style="opacity:.6;font-size:.85em">'
                        f'{_zeit(s["start"])} – {_zeit(s["ende"])}</span>'
                        f'<div style="margin-top:4px">{s["text"]}</div></div>')
        return bloecke or "<p>Keine Sprache erkannt.</p>"

    if use_case == USE_CASES[2]:
        gesamt = sum(st["dauer"] for st in daten["stats"].values()) or 1.0
        zeilen = ""
        for lb, st in sorted(daten["stats"].items(),
                             key=lambda x: x[1]["dauer"], reverse=True):
            anteil = 100 * st["dauer"] / gesamt
            zeilen += (f'<div style="margin:12px 0">{_chip(namen[lb], farben[lb])} '
                       f'{_zeit(st["dauer"])} min ({anteil:.0f} %), '
                       f'{st["turns"]} Redebeiträge'
                       f'<div style="height:14px;background:rgba(128,128,128,.15);'
                       f'border-radius:7px;margin-top:4px"><div style="height:14px;'
                       f'width:{anteil:.1f}%;background:{farben[lb]};'
                       f'border-radius:7px"></div></div></div>')
        return (f"<p>Aufnahme: {_zeit(daten['dauer'])} min &nbsp;·&nbsp; "
                f"{len(namen)} Sprecher</p>" + zeilen)

    if use_case == USE_CASES[3]:
        if not daten["overlaps"]:
            return "<p>✅ Kein Durcheinanderreden gefunden.</p>"
        zeilen = ""
        for o in daten["overlaps"]:
            chips = " ".join(_chip(namen[lb], farben[lb]) for lb in o["wer"])
            zeilen += (f'<tr><td style="padding:4px 12px 4px 0;white-space:nowrap">'
                       f'{_zeit(o["start"])} – {_zeit(o["ende"])}</td>'
                       f'<td style="padding:4px">{chips}</td></tr>')
        gesamt = sum(o["ende"] - o["start"] for o in daten["overlaps"])
        return (f"<p>⚠️ {len(daten['overlaps'])} Stellen mit gleichzeitigem "
                f"Sprechen (insgesamt {gesamt:.1f} s):</p>"
                f'<table style="border-collapse:collapse">{zeilen}</table>')

    legende = " ".join(_chip(namen[lb], farben[lb]) for lb in sorted(namen))
    zeilen = ""
    for s in daten["segmente"]:
        zeilen += (f'<tr><td style="padding:4px 12px 4px 0;white-space:nowrap">'
                   f'{_zeit(s["start"])} – {_zeit(s["ende"])}</td>'
                   f'<td style="padding:4px">'
                   f'{_chip(namen[s["label"]], farben[s["label"]])}</td></tr>')
    return (f"<p>{legende}</p>" + _zeitleiste(daten, namen, farben) +
            f'<table style="border-collapse:collapse">{zeilen}</table>')


# ---------- Gradio-Handler ----------

def analysieren(audio, use_case, num_speakers):
    if audio is None:
        return "<p>Bitte zuerst Audio aufnehmen oder eine Datei hochladen.</p>", ""
    try:
        daten = _analyse_gpu(audio, use_case == USE_CASES[1], num_speakers)
    except Exception as e:
        return f"<p>❌ Fehler bei der Analyse: {e}</p>", ""
    zeitpunkt = datetime.now(ZEITZONE)
    md_text = _markdown(daten, use_case, os.path.basename(audio), zeitpunkt)
    try:
        pfad = _speichern(md_text, zeitpunkt)
        status = f"💾 Gespeichert: `{pfad}` in [{DATEN_REPO}](https://huggingface.co/datasets/{DATEN_REPO})"
    except Exception as e:
        status = f"⚠️ Analyse ok, aber Speichern fehlgeschlagen: {e}"
    return _html(daten, use_case), status


def _frontmatter(text):
    if text.startswith("---"):
        try:
            ende = text.index("\n---", 3)
            return (yaml.safe_load(text[3:ende]) or {}), text[ende + 4:]
        except (ValueError, yaml.YAMLError):
            pass
    return {}, text


def dashboard_laden():
    dateien = sorted(
        (f for f in api.list_repo_files(DATEN_REPO, repo_type="dataset")
         if f.startswith("aufnahmen/") and f.endswith(".md")), reverse=True)[:20]
    zeilen, pfade = [], []
    for f in dateien:
        lokal = hf_hub_download(DATEN_REPO, f, repo_type="dataset", token=HF_TOKEN)
        with open(lokal, encoding="utf-8") as fh:
            fm, _ = _frontmatter(fh.read())
        datum = str(fm.get("datum", ""))[:16].replace("T", " ")
        zeilen.append([datum, fm.get("titel", f),
                       fm.get("anwendungsfall", "?"), fm.get("sprecher", "?"),
                       _zeit(fm.get("dauer_s", 0))])
        pfade.append(f)
    if not zeilen:
        zeilen = [["–", "Noch keine Aufzeichnungen", "", "", ""]]
    return zeilen, pfade


def eintrag_anzeigen(pfade, evt: gr.SelectData):
    idx = evt.index[0]
    if not pfade or idx >= len(pfade):
        return "*Kein Eintrag ausgewählt.*", None
    lokal = hf_hub_download(DATEN_REPO, pfade[idx], repo_type="dataset",
                            token=HF_TOKEN)
    with open(lokal, encoding="utf-8") as fh:
        text = fh.read()
    fm, body = _frontmatter(text)
    vorschau = (f"```yaml\n{yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()}\n```\n"
                + body)
    return vorschau, lokal


with gr.Blocks(theme=gr.themes.Soft(), title="Sprecher-Analyse") as demo:
    gr.Markdown("# 🎙️ Sprecher-Analyse")
    with gr.Tabs():
        with gr.Tab("Analyse"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio = gr.Audio(sources=["microphone", "upload"],
                                     type="filepath", label="Audio")
                    use_case = gr.Radio(USE_CASES, value=USE_CASES[0],
                                        label="Anwendungsfall")
                    num_speakers = gr.Number(
                        value=0, precision=0,
                        label="Anzahl Sprecher (0 = automatisch)")
                    start = gr.Button("Analysieren", variant="primary")
                    status = gr.Markdown()
                with gr.Column(scale=2):
                    ergebnis = gr.HTML(label="Ergebnis")
        with gr.Tab("📋 Dashboard"):
            aktualisieren = gr.Button("🔄 Aktualisieren")
            tabelle = gr.Dataframe(
                headers=["Datum", "Titel", "Anwendungsfall", "Sprecher", "Dauer"],
                interactive=False, wrap=True)
            pfade_state = gr.State([])
            with gr.Row():
                with gr.Column(scale=2):
                    vorschau = gr.Markdown("*Zeile antippen für die Vorschau.*")
                with gr.Column(scale=1):
                    datei = gr.File(label="Markdown-Datei")

    start.click(analysieren, [audio, use_case, num_speakers], [ergebnis, status])
    aktualisieren.click(dashboard_laden, None, [tabelle, pfade_state])
    tabelle.select(eintrag_anzeigen, [pfade_state], [vorschau, datei])
    demo.load(dashboard_laden, None, [tabelle, pfade_state])

demo.launch(pwa=True)
