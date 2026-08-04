"""Smoke-Test für pyannote.audio ohne Hugging-Face-Token.

Setup:
    python3 -m venv venv && venv/bin/pip install pyannote.audio
    apt-get install -y ffmpeg   # für torchcodec-Audiodekodierung
Ausführen:
    venv/bin/python test_pyannote.py
"""
import torch

print("== 1. Versionen ==")
import pyannote.audio
import pyannote.core
print(f"pyannote.audio {pyannote.audio.__version__}")
print(f"torch {torch.__version__} (CUDA verfügbar: {torch.cuda.is_available()})")

print("\n== 2. pyannote.core: Segmente & Annotationen ==")
from pyannote.core import Segment, Annotation, Timeline
ann = Annotation()
ann[Segment(0.0, 2.5)] = "sprecher_A"
ann[Segment(2.0, 5.0)] = "sprecher_B"
ann[Segment(5.5, 7.0)] = "sprecher_A"
print(f"Labels: {ann.labels()}")
print(f"Sprechdauer A: {ann.label_duration('sprecher_A'):.1f}s")
overlap = ann.get_overlap()
print(f"Überlappung: {overlap}")

print("\n== 3. Diarization Error Rate (Metrik) ==")
from pyannote.metrics.diarization import DiarizationErrorRate
hyp = Annotation()
hyp[Segment(0.0, 2.4)] = "spk1"
hyp[Segment(2.4, 5.0)] = "spk2"
hyp[Segment(5.5, 7.0)] = "spk1"
der = DiarizationErrorRate()(ann, hyp)
print(f"DER Referenz vs. Hypothese: {der:.1%}")

print("\n== 4. Segmentierungsmodell (PyanNet, zufällige Gewichte) ==")
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.core.task import Problem, Resolution, Specifications
model = PyanNet(sample_rate=16000, num_channels=1)
model.specifications = Specifications(
    problem=Problem.MULTI_LABEL_CLASSIFICATION,
    resolution=Resolution.FRAME,
    duration=5.0,
    classes=["sprecher_1", "sprecher_2", "sprecher_3"],
)
model.build()
n_params = sum(p.numel() for p in model.parameters())
print(f"Modell instanziiert: {n_params:,} Parameter")

# 5 Sekunden synthetisches Audio (Sinuston + Rauschen)
sr = 16000
t = torch.arange(sr * 5) / sr
audio = (0.3 * torch.sin(2 * 3.14159 * 440 * t) + 0.05 * torch.randn(sr * 5))
waveform = audio.reshape(1, 1, -1)  # (batch, channel, samples)
with torch.no_grad():
    out = model(waveform)
print(f"Forward-Pass OK: Input {tuple(waveform.shape)} -> Output {tuple(out.shape)}")

print("\n== 5. Audiodatei einlesen (torchcodec/FFmpeg) ==")
import subprocess, os
wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_tone.wav")
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
     "-i", "sine=frequency=440:duration=3", "-ar", "16000", "-ac", "1", wav],
    check=True,
)
from pyannote.audio.core.io import Audio
loader = Audio(sample_rate=16000, mono="downmix")
wave, sample_rate = loader(wav)
print(f"WAV geladen: shape {tuple(wave.shape)}, {sample_rate} Hz, Dauer {loader.get_duration(wav):.1f}s")

print("\nAlle Tests bestanden.")
