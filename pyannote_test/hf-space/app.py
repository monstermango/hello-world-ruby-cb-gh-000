"""Gradio-App für Speaker Diarization mit pyannote (HF Space auf ZeroGPU)."""
import os

import gradio as gr
import spaces
import torch
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"],
)
pipeline.to(torch.device("cuda"))


@spaces.GPU(duration=120)
def diarize(audio, num_speakers):
    if audio is None:
        return "Bitte zuerst Audio aufnehmen oder eine Datei hochladen."
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = pipeline(audio, **kwargs)
    diarization = getattr(output, "speaker_diarization", output)
    lines = ["Start      Ende      Sprecher", "-" * 32]
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        lines.append(f"{turn.start:7.1f}s {turn.end:8.1f}s  {speaker}")
    lines.append("")
    lines.append(f"Erkannte Sprecher: {len(diarization.labels())}")
    return "\n".join(lines)


demo = gr.Interface(
    fn=diarize,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
        gr.Number(value=0, precision=0, label="Anzahl Sprecher (0 = automatisch)"),
    ],
    outputs=gr.Textbox(label="Ergebnis", lines=14),
    title="Speaker Diarization (pyannote)",
    description="Audio aufnehmen oder hochladen — wer spricht wann?",
    flagging_mode="never",
)

demo.launch()
