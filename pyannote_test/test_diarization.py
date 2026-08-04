"""Test der vortrainierten pyannote-Diarization-Pipeline (Token aus HF_TOKEN)."""
import os, sys, time
import functools
import torch

# pyannote-3.x-Checkpoints (offizielle HF-Repos) sind mit torch>=2.6 nur mit
# weights_only=False ladbar
torch.load = functools.partial(torch.load, weights_only=False)

from pyannote.audio import Pipeline

token = os.environ["HF_TOKEN"]
audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dialog.wav")

candidates = [
    "pyannote/speaker-diarization-community-1",  # Flaggschiff für pyannote.audio 4.x
    "pyannote/speaker-diarization-3.1",
]

pipeline = None
for name in candidates:
    try:
        print(f"Lade Pipeline: {name} ...")
        t0 = time.time()
        pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
        if pipeline is None:
            print("  Zugriff verweigert (None zurückgegeben)\n")
            continue
        print(f"  geladen in {time.time()-t0:.1f}s")
        break
    except Exception as e:
        print(f"  FEHLER: {type(e).__name__}: {e}\n")

if pipeline is None:
    sys.exit("Keine Pipeline konnte geladen werden.")

print(f"\nDiarization von {audio_file} (CPU) ...")
t0 = time.time()
result = pipeline(audio_file)
elapsed = time.time() - t0
print(f"Fertig in {elapsed:.1f}s\n")

diarization = getattr(result, "speaker_diarization", result)
print("Erkannte Sprecher-Segmente:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"  {turn.start:6.2f}s – {turn.end:6.2f}s  {speaker}")

print(f"\nAnzahl erkannter Sprecher: {len(diarization.labels())}")
