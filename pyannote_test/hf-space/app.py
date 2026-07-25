"""Sprecher-Analyse: ein Ablauf, vollständiges Ergebnis (HF Space auf ZeroGPU)."""
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


def _zeit(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d}"


def _namen_farben(labels):
    namen = {lb: f"Sprecher {i + 1}" for i, lb in enumerate(sorted(labels))}
    farben = {lb: FARBEN[i % len(FARBEN)] for i, lb in enumerate(sorted(labels))}
    return namen, farben


@spaces.GPU(duration=120)
def _analyse_gpu(audio_path, num_speakers):
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = diarizer(audio_path, **kwargs)
    dia = getattr(output, "speaker_diarization", output)

    segmente = []
    for turn, _, lb in dia.itertracks(yield_label=True):
        text = None
        if turn.duration >= 0.3:
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

def _markdown(daten, quelle, zeitpunkt):
    namen, _ = _namen_farben(daten["stats"])
    frontmatter = {
        "titel": f"Aufnahme {zeitpunkt:%Y-%m-%d %H:%M}",
        "datum": zeitpunkt.isoformat(timespec="seconds"),
        "dauer_s": daten["dauer"],
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


# ---------- Ergebnis-Darstellung ----------

def _chip(text, farbe):
    return (f'<span style="background:{farbe};color:#fff;border-radius:12px;'
            f'padding:2px 10px;font-weight:600;font-size:.85em;'
            f'white-space:nowrap">{text}</span>')


def _html(daten, zeitpunkt):
    namen, farben = _namen_farben(daten["stats"])
    gesamt = sum(st["dauer"] for st in daten["stats"].values()) or 1.0
    dauer = daten["dauer"] or 1.0

    kopf = (f'<div style="display:flex;gap:16px;flex-wrap:wrap;opacity:.8;'
            f'font-size:.9em;margin-bottom:12px">'
            f'<span>🕒 {_zeit(daten["dauer"])} min</span>'
            f'<span>👥 {len(namen)} Sprecher</span>'
            f'<span>📅 {zeitpunkt:%d.%m.%Y %H:%M}</span></div>')

    anteil_bloecke, legende = "", ""
    for lb in sorted(namen, key=lambda x: daten["stats"][x]["dauer"], reverse=True):
        st = daten["stats"][lb]
        anteil = 100 * st["dauer"] / gesamt
        anteil_bloecke += (f'<div style="width:{anteil:.1f}%;'
                           f'background:{farben[lb]}" title="{namen[lb]}"></div>')
        legende += (f'<span style="margin-right:12px;white-space:nowrap">'
                    f'{_chip(namen[lb], farben[lb])} '
                    f'<span style="font-size:.85em;opacity:.8">{anteil:.0f} % '
                    f'· {_zeit(st["dauer"])} min</span></span>')
    anteile = (f'<div style="display:flex;height:16px;border-radius:8px;'
               f'overflow:hidden;margin:4px 0 8px 0">{anteil_bloecke}</div>'
               f'<div style="line-height:2">{legende}</div>')

    lanes = ""
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
        lanes += (f'<div style="position:relative;height:14px;margin:3px 0;'
                  f'background:rgba(128,128,128,.12);border-radius:3px">'
                  f'{bloecke}</div>')
    zeitleiste = (f'<div style="margin:16px 0 4px 0">{lanes}'
                  f'<div style="display:flex;justify-content:space-between;'
                  f'font-size:.75em;opacity:.6"><span>0:00</span>'
                  f'<span>{_zeit(dauer)}</span></div></div>')

    protokoll = '<h3 style="margin:20px 0 8px 0">Gespräch</h3>'
    for s in daten["segmente"]:
        if not s["text"]:
            continue
        protokoll += (
            f'<div style="display:flex;gap:10px;margin:10px 0">'
            f'<div style="flex:0 0 4px;border-radius:2px;'
            f'background:{farben[s["label"]]}"></div>'
            f'<div style="min-width:0"><div style="font-size:.78em;opacity:.65;'
            f'margin-bottom:2px">{namen[s["label"]]} · '
            f'{_zeit(s["start"])}–{_zeit(s["ende"])}</div>'
            f'<div style="background:rgba(128,128,128,.1);padding:8px 12px;'
            f'border-radius:10px">{s["text"]}</div></div></div>')

    hinweis = ""
    if daten["overlaps"]:
        stellen = ", ".join(f"{_zeit(o['start'])}–{_zeit(o['ende'])}"
                            for o in daten["overlaps"])
        hinweis = (f'<div style="margin-top:14px;padding:8px 12px;'
                   f'border-radius:8px;background:rgba(240,173,78,.15);'
                   f'font-size:.85em">⚠️ Gleichzeitiges Sprechen bei: '
                   f'{stellen}</div>')

    return kopf + anteile + zeitleiste + protokoll + hinweis


# ---------- Gradio-Handler ----------

def analysieren(audio, num_speakers):
    if audio is None:
        return "<p>Bitte zuerst Audio aufnehmen oder eine Datei hochladen.</p>", ""
    try:
        daten = _analyse_gpu(audio, num_speakers)
    except Exception as e:
        return f"<p>❌ Fehler bei der Analyse: {e}</p>", ""
    zeitpunkt = datetime.now(ZEITZONE)
    md_text = _markdown(daten, os.path.basename(audio), zeitpunkt)
    try:
        pfad = _speichern(md_text, zeitpunkt)
        status = (f"💾 Gespeichert: [`{pfad}`]"
                  f"(https://huggingface.co/datasets/{DATEN_REPO})")
    except Exception as e:
        status = f"⚠️ Analyse ok, aber Speichern fehlgeschlagen: {e}"
    return _html(daten, zeitpunkt), status


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
        zeilen.append([datum, fm.get("titel", f), fm.get("sprecher", "?"),
                       _zeit(fm.get("dauer_s", 0))])
        pfade.append(f)
    if not zeilen:
        zeilen = [["–", "Noch keine Aufzeichnungen", "", ""]]
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


CSS = """
footer {display: none !important}
.gradio-container {max-width: 760px !important; margin: 0 auto !important}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Sprecher-Analyse", css=CSS) as demo:
    gr.Markdown("# 🎙️ Sprecher-Analyse")
    with gr.Tabs():
        with gr.Tab("Analyse"):
            audio = gr.Audio(sources=["microphone", "upload"],
                             type="filepath", label="Audio")
            with gr.Accordion("Optionen", open=False):
                num_speakers = gr.Number(
                    value=0, precision=0,
                    label="Anzahl Sprecher (0 = automatisch)")
            start = gr.Button("Analysieren", variant="primary", size="lg")
            status = gr.Markdown()
            ergebnis = gr.HTML()
        with gr.Tab("📋 Verlauf"):
            aktualisieren = gr.Button("🔄 Aktualisieren", size="sm")
            tabelle = gr.Dataframe(
                headers=["Datum", "Titel", "Sprecher", "Dauer"],
                interactive=False, wrap=True)
            pfade_state = gr.State([])
            vorschau = gr.Markdown("*Zeile antippen für die Vorschau.*")
            datei = gr.File(label="Markdown-Datei")

    start.click(analysieren, [audio, num_speakers], [ergebnis, status])
    aktualisieren.click(dashboard_laden, None, [tabelle, pfade_state])
    tabelle.select(eintrag_anzeigen, [pfade_state], [vorschau, datei])
    demo.load(dashboard_laden, None, [tabelle, pfade_state])

demo.launch(
    pwa=True,
    ssr_mode=False,
    auth=(os.environ["APP_USER"], os.environ["APP_PASS"]),
    auth_message="Bitte mit Benutzername und Passwort anmelden.",
)
