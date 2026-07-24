# pyannote-Smoke-Test — Ergebnis (2026-07-24)

Getestet wurde `pyannote.audio` in der Remote-Umgebung (Python 3.11.15, CPU, kein Hugging-Face-Token).

## Ergebnis: alle Tests bestanden

```
== 1. Versionen ==
pyannote.audio 4.0.7
torch 2.13.0+cu130 (CUDA verfügbar: False)

== 2. pyannote.core: Segmente & Annotationen ==
Labels: ['sprecher_A', 'sprecher_B']
Sprechdauer A: 4.0s
Überlappung: [[ 00:00:02.000 -->  00:00:02.500]]

== 3. Diarization Error Rate (Metrik) ==
DER Referenz vs. Hypothese: 7.1%

== 4. Segmentierungsmodell (PyanNet, zufällige Gewichte) ==
Modell instanziiert: 682,221 Parameter
Forward-Pass OK: Input (1, 1, 80000) -> Output (1, 293, 3)

== 5. Audiodatei einlesen (torchcodec/FFmpeg) ==
WAV geladen: shape (1, 48000), 16000 Hz, Dauer 3.0s

Alle Tests bestanden.
```

## Hinweise

- **FFmpeg musste nachinstalliert werden** (`apt-get install -y ffmpeg`), sonst kann
  torchcodec keine Audiodateien dekodieren (`libavutil.so` fehlt).
- **Vortrainierte Modelle** (z. B. `pyannote/speaker-diarization-3.1`) sind auf
  Hugging Face zugangsbeschränkt und benötigen ein Token (`HF_TOKEN`). Der Test
  deckt daher Installation, Kern-Datenstrukturen, Metriken, einen Modell-Forward-Pass
  mit zufälligen Gewichten und die Audio-Ein-/Ausgabe ab — nicht die eigentliche
  Diarization-Qualität.
- Installation zieht ~3 GB Abhängigkeiten (PyTorch inkl. CUDA-Bibliotheken, obwohl
  hier nur CPU verfügbar ist).
