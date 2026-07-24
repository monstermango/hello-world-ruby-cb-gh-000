"""Sprecher-Analyse: Gradio-App mit Standard-Use-Cases (HF Space auf ZeroGPU)."""
import os

import gradio as gr
import spaces
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from transformers import pipeline as hf_pipeline

HF_TOKEN = os.environ["HF_TOKEN"]

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


def _diarize(audio_path, num_speakers):
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = diarizer(audio_path, **kwargs)
    dia = getattr(output, "speaker_diarization", output)
    labels = sorted(dia.labels())
    namen = {lb: f"Sprecher {i + 1}" for i, lb in enumerate(labels)}
    farben = {lb: FARBEN[i % len(FARBEN)] for i, lb in enumerate(labels)}
    return dia, namen, farben


def _chip(text, farbe):
    return (f'<span style="background:{farbe};color:#fff;border-radius:12px;'
            f'padding:2px 10px;font-weight:600;white-space:nowrap">{text}</span>')


def _zeitleiste(dia, farben, dauer):
    lanes = []
    for lb in sorted(dia.labels()):
        bloecke = ""
        for seg in dia.label_timeline(lb):
            links = 100 * seg.start / dauer
            breite = max(0.5, 100 * seg.duration / dauer)
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


def uc_wer_wann(audio_path, num_speakers):
    dia, namen, farben = _diarize(audio_path, num_speakers)
    dauer = loader.get_duration(audio_path)
    legende = " ".join(_chip(namen[lb], farben[lb]) for lb in sorted(dia.labels()))
    zeilen = ""
    for turn, _, lb in dia.itertracks(yield_label=True):
        zeilen += (f'<tr><td style="padding:4px 12px 4px 0;white-space:nowrap">'
                   f'{_zeit(turn.start)} – {_zeit(turn.end)}</td>'
                   f'<td style="padding:4px">{_chip(namen[lb], farben[lb])}</td></tr>')
    return (f"<p>{legende}</p>" + _zeitleiste(dia, farben, dauer) +
            f'<table style="border-collapse:collapse">{zeilen}</table>')


def uc_protokoll(audio_path, num_speakers):
    dia, namen, farben = _diarize(audio_path, num_speakers)
    bloecke = ""
    for turn, _, lb in dia.itertracks(yield_label=True):
        if turn.duration < 0.3:
            continue
        wellenform, sr = loader.crop(audio_path, Segment(turn.start, turn.end))
        text = asr({"array": wellenform.squeeze(0).numpy(),
                    "sampling_rate": sr})["text"].strip()
        bloecke += (f'<div style="margin:10px 0;padding-left:12px;'
                    f'border-left:4px solid {farben[lb]}">'
                    f'{_chip(namen[lb], farben[lb])} '
                    f'<span style="opacity:.6;font-size:.85em">'
                    f'{_zeit(turn.start)} – {_zeit(turn.end)}</span>'
                    f'<div style="margin-top:4px">{text}</div></div>')
    return bloecke or "<p>Keine Sprache erkannt.</p>"


def uc_statistik(audio_path, num_speakers):
    dia, namen, farben = _diarize(audio_path, num_speakers)
    gesamt = sum(dia.label_duration(lb) for lb in dia.labels()) or 1.0
    zeilen = ""
    for lb in sorted(dia.labels(), key=dia.label_duration, reverse=True):
        d = dia.label_duration(lb)
        anteil = 100 * d / gesamt
        turns = len(dia.label_timeline(lb))
        zeilen += (f'<div style="margin:12px 0">{_chip(namen[lb], farben[lb])} '
                   f'{_zeit(d)} min ({anteil:.0f} %), {turns} Redebeiträge'
                   f'<div style="height:14px;background:rgba(128,128,128,.15);'
                   f'border-radius:7px;margin-top:4px"><div style="height:14px;'
                   f'width:{anteil:.1f}%;background:{farben[lb]};'
                   f'border-radius:7px"></div></div></div>')
    dauer = loader.get_duration(audio_path)
    kopf = (f"<p>Aufnahme: {_zeit(dauer)} min &nbsp;·&nbsp; "
            f"Sprechzeit gesamt: {_zeit(gesamt)} min &nbsp;·&nbsp; "
            f"{len(dia.labels())} Sprecher</p>")
    return kopf + zeilen


def uc_overlap(audio_path, num_speakers):
    dia, namen, farben = _diarize(audio_path, num_speakers)
    overlap = dia.get_overlap()
    if not overlap:
        return "<p>✅ Kein Durcheinanderreden gefunden.</p>"
    gesamt = sum(seg.duration for seg in overlap)
    zeilen = ""
    for seg in overlap:
        beteiligte = [lb for lb in dia.labels()
                      if dia.label_timeline(lb).crop(seg)]
        chips = " ".join(_chip(namen[lb], farben[lb]) for lb in sorted(beteiligte))
        zeilen += (f'<tr><td style="padding:4px 12px 4px 0;white-space:nowrap">'
                   f'{_zeit(seg.start)} – {_zeit(seg.end)}</td>'
                   f'<td style="padding:4px">{chips}</td></tr>')
    return (f"<p>⚠️ {len(overlap)} Stellen mit gleichzeitigem Sprechen "
            f"(insgesamt {gesamt:.1f} s):</p>"
            f'<table style="border-collapse:collapse">{zeilen}</table>')


@spaces.GPU(duration=120)
def analysieren(audio, use_case, num_speakers):
    if audio is None:
        return "<p>Bitte zuerst Audio aufnehmen oder eine Datei hochladen.</p>"
    try:
        fn = {
            USE_CASES[0]: uc_wer_wann,
            USE_CASES[1]: uc_protokoll,
            USE_CASES[2]: uc_statistik,
            USE_CASES[3]: uc_overlap,
        }[use_case]
        return fn(audio, num_speakers)
    except Exception as e:
        return f"<p>❌ Fehler bei der Analyse: {e}</p>"


with gr.Blocks(theme=gr.themes.Soft(), title="Sprecher-Analyse") as demo:
    gr.Markdown("# 🎙️ Sprecher-Analyse\n"
                "Audio aufnehmen oder hochladen und einen Anwendungsfall wählen.")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"],
                             type="filepath", label="Audio")
            use_case = gr.Radio(USE_CASES, value=USE_CASES[0],
                                label="Anwendungsfall")
            num_speakers = gr.Number(value=0, precision=0,
                                     label="Anzahl Sprecher (0 = automatisch)")
            start = gr.Button("Analysieren", variant="primary")
        with gr.Column(scale=2):
            ergebnis = gr.HTML(label="Ergebnis")
    start.click(analysieren, [audio, use_case, num_speakers], ergebnis)

demo.launch()
