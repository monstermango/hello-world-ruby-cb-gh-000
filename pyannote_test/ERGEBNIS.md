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

## Teil 2: Vortrainierte Diarization-Pipeline (mit HF-Token)

Mit einem Hugging-Face-Token wurde zusätzlich `pyannote/speaker-diarization-3.1`
auf einem synthetisch erzeugten 23-Sekunden-Dialog (zwei espeak-ng-Stimmen,
vier Sprecherwechsel) getestet — **erfolgreich**:

```
Diarization von dialog.wav (CPU) ...
Fertig in 17.0s

Erkannte Sprecher-Segmente:
    0.03s –   6.06s  SPEAKER_00
    7.05s –  12.67s  SPEAKER_01
   13.56s –  19.25s  SPEAKER_00
   20.23s –  23.13s  SPEAKER_01

Anzahl erkannter Sprecher: 2
```

Alle vier Sprecherwechsel und beide Sprecher wurden korrekt erkannt.
Laufzeit auf CPU: ca. 0,7× Echtzeit (17 s für 23,4 s Audio).

### Benötigtes Setup für die vortrainierte Pipeline

- `pyannote/speaker-diarization-community-1` (Standard in pyannote.audio 4.x) war
  für den Token **nicht freigeschaltet** (HTTP 403) — und auch die 3.1-Pipeline
  lädt unter 4.x eine Datei aus diesem Repo. Deshalb Ausweichlösung:
  **pyannote.audio 3.3.2** mit gepinnten Abhängigkeiten:
  `torch==2.7.1` (CPU), `torchaudio==2.7.1`, `huggingface_hub==0.25.2`,
  `soundfile`, `matplotlib`.
- Für torch ≥ 2.6 muss `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` gesetzt werden,
  da die pyannote-3.x-Checkpoints sonst am neuen `weights_only=True`-Default
  scheitern.
- Token als `HF_TOKEN`-Umgebungsvariable übergeben (nie einchecken).

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
