"""Sprecher-Analyse: mobil-optimierte App (HF Space auf ZeroGPU)."""
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
    return (f'<span style="background:{farbe};color:#fff;border-radius:999px;'
            f'padding:2px 10px;font-weight:600;font-size:.85rem;'
            f'white-space:nowrap">{text}</span>')


def _html(daten, zeitpunkt):
    namen, farben = _namen_farben(daten["stats"])
    gesamt = sum(st["dauer"] for st in daten["stats"].values()) or 1.0
    dauer = daten["dauer"] or 1.0

    kopf = (f'<div class="sa-meta">'
            f'<span>🕒 {_zeit(daten["dauer"])} min</span>'
            f'<span>👥 {len(namen)} Sprecher</span>'
            f'<span>📅 {zeitpunkt:%d.%m.%Y %H:%M}</span></div>')

    anteil_bloecke, legende = "", ""
    for lb in sorted(namen, key=lambda x: daten["stats"][x]["dauer"], reverse=True):
        st = daten["stats"][lb]
        anteil = 100 * st["dauer"] / gesamt
        anteil_bloecke += (f'<div style="width:{anteil:.1f}%;'
                           f'background:{farben[lb]}"></div>')
        legende += (f'<span class="sa-leg">{_chip(namen[lb], farben[lb])} '
                    f'<span class="sa-dim">{anteil:.0f} % · '
                    f'{_zeit(st["dauer"])} min</span></span>')
    anteile = (f'<div class="sa-bar">{anteil_bloecke}</div>'
               f'<div class="sa-legs">{legende}</div>')

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
        lanes += f'<div class="sa-lane">{bloecke}</div>'
    zeitleiste = (f'<div class="sa-timeline">{lanes}'
                  f'<div class="sa-ticks"><span>0:00</span>'
                  f'<span>{_zeit(dauer)}</span></div></div>')

    protokoll = '<h3 class="sa-h">Gespräch</h3>'
    for s in daten["segmente"]:
        if not s["text"]:
            continue
        protokoll += (
            f'<div class="sa-msg">'
            f'<div class="sa-rail" style="background:{farben[s["label"]]}"></div>'
            f'<div class="sa-msg-body"><div class="sa-msg-head">'
            f'{namen[s["label"]]} · {_zeit(s["start"])}–{_zeit(s["ende"])}</div>'
            f'<div class="sa-bubble">{s["text"]}</div></div></div>')

    hinweis = ""
    if daten["overlaps"]:
        stellen = ", ".join(f"{_zeit(o['start'])}–{_zeit(o['ende'])}"
                            for o in daten["overlaps"])
        hinweis = (f'<div class="sa-warn">⚠️ Gleichzeitiges Sprechen bei: '
                   f'{stellen}</div>')

    return (f'<div class="sa-result">{kopf}{anteile}{zeitleiste}'
            f'{protokoll}{hinweis}</div>')


# ---------- Gradio-Handler ----------

def analysieren(audio, num_speakers):
    if audio is None:
        return ('<div class="sa-result">Bitte zuerst Audio aufnehmen '
                'oder eine Datei hochladen.</div>'), ""
    try:
        daten = _analyse_gpu(audio, num_speakers)
    except Exception as e:
        return f'<div class="sa-result">❌ Fehler bei der Analyse: {e}</div>', ""
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


def verlauf_laden():
    dateien = sorted(
        (f for f in api.list_repo_files(DATEN_REPO, repo_type="dataset")
         if f.startswith("aufnahmen/") and f.endswith(".md")), reverse=True)[:20]
    karten, auswahl = "", []
    for f in dateien:
        lokal = hf_hub_download(DATEN_REPO, f, repo_type="dataset", token=HF_TOKEN)
        with open(lokal, encoding="utf-8") as fh:
            fm, _ = _frontmatter(fh.read())
        datum = str(fm.get("datum", ""))[:16].replace("T", " ")
        titel = fm.get("titel", f)
        karten += (f'<div class="sa-card"><div class="sa-card-title">{titel}</div>'
                   f'<div class="sa-card-meta">📅 {datum} &nbsp; '
                   f'👥 {fm.get("sprecher", "?")} &nbsp; '
                   f'🕒 {_zeit(fm.get("dauer_s", 0))} min</div></div>')
        auswahl.append((f"{titel}", f))
    if not karten:
        karten = '<div class="sa-card">Noch keine Aufzeichnungen.</div>'
    return karten, gr.update(choices=auswahl, value=None)


def eintrag_anzeigen(pfad):
    if not pfad:
        return "", None
    lokal = hf_hub_download(DATEN_REPO, pfad, repo_type="dataset", token=HF_TOKEN)
    with open(lokal, encoding="utf-8") as fh:
        text = fh.read()
    fm, body = _frontmatter(text)
    vorschau = (f"```yaml\n{yaml.safe_dump(fm, allow_unicode=True, sort_keys=False).strip()}\n```\n"
                + body)
    return vorschau, lokal


CSS = """
footer {display: none !important}
.gradio-container {max-width: 680px !important; margin: 0 auto !important;
  padding: clamp(8px, 3vw, 24px) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", sans-serif !important}
.gradio-container h1 {font-size: clamp(1.3rem, 5vw, 1.8rem); margin: 4px 0 12px 0}
button.primary {width: 100%; min-height: 52px; font-size: 1.05rem;
  border-radius: 14px}
.tab-nav button {min-height: 44px; font-size: 1rem}

.sa-result {font-size: 1rem; line-height: 1.5; overflow-wrap: anywhere}
.sa-meta {display: flex; gap: 14px; flex-wrap: wrap; opacity: .75;
  font-size: .9rem; margin: 4px 0 14px 0}
.sa-bar {display: flex; height: 16px; border-radius: 999px; overflow: hidden;
  margin: 4px 0 10px 0}
.sa-legs {line-height: 2.2}
.sa-leg {margin-right: 14px; white-space: nowrap}
.sa-dim {font-size: .85rem; opacity: .75}
.sa-timeline {margin: 18px 0 6px 0}
.sa-lane {position: relative; height: 14px; margin: 4px 0;
  background: rgba(128,128,128,.12); border-radius: 3px}
.sa-ticks {display: flex; justify-content: space-between; font-size: .75rem;
  opacity: .6}
.sa-h {margin: 22px 0 6px 0; font-size: 1.1rem}
.sa-msg {display: flex; gap: 10px; margin: 12px 0}
.sa-rail {flex: 0 0 4px; border-radius: 2px}
.sa-msg-body {min-width: 0; flex: 1}
.sa-msg-head {font-size: .78rem; opacity: .65; margin-bottom: 3px}
.sa-bubble {background: rgba(128,128,128,.1); padding: 10px 14px;
  border-radius: 12px; overflow-wrap: anywhere}
.sa-warn {margin-top: 16px; padding: 10px 14px; border-radius: 10px;
  background: rgba(240,173,78,.15); font-size: .9rem}

.sa-card {padding: 12px 14px; border-radius: 12px;
  background: rgba(128,128,128,.08); margin: 8px 0}
.sa-card-title {font-weight: 600}
.sa-card-meta {font-size: .85rem; opacity: .7; margin-top: 2px}

@media (max-width: 640px) {
  .gradio-container {padding: 8px !important}
  .block {padding: 8px !important}
}
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
            start = gr.Button("Analysieren", variant="primary")
            status = gr.Markdown()
            ergebnis = gr.HTML()
        with gr.Tab("Verlauf"):
            verlauf_html = gr.HTML()
            eintrag = gr.Dropdown(label="Eintrag öffnen", choices=[],
                                  interactive=True)
            vorschau = gr.Markdown()
            datei = gr.File(label="Markdown-Datei")

    start.click(analysieren, [audio, num_speakers], [ergebnis, status]) \
         .then(verlauf_laden, None, [verlauf_html, eintrag])
    eintrag.change(eintrag_anzeigen, [eintrag], [vorschau, datei])
    demo.load(verlauf_laden, None, [verlauf_html, eintrag])

demo.launch(
    pwa=True,
    ssr_mode=False,
    auth=(os.environ["APP_USER"], os.environ["APP_PASS"]),
    auth_message="Bitte mit Benutzername und Passwort anmelden.",
)
