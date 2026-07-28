"""API-Backend der Sprecher-Analyse (HF Space auf ZeroGPU).

Die eigentliche App (PWA) läuft als eigener Static Space und spricht die
hier registrierten API-Endpunkte an. Jeder Endpunkt verlangt den
Zugangsschlüssel (Secret APP_PASS); erst nach der Prüfung wird GPU-Zeit
verbraucht.
"""
import hmac
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
import traceback
import unicodedata
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import spaces
import torch
import torchaudio.functional as AF
import yaml
from torchaudio.pipelines import MMS_FA as FA_BUNDLE
from huggingface_hub import HfApi, hf_hub_download
from pyannote.audio import Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
from transformers import pipeline as hf_pipeline

from kern import (FA_ANTEIL, FA_BLOCK, FA_FENSTER, FARBEN, FENSTER,
                  GLOSSAR_DATEI, GLOSSAR_MAX, GLOSSAR_PROMPT,
                  _begriffe_zaehlen, _decode_optionen, _fenstergrenzen,
                  _korrekturziele, _korrigieren,
                  _frontmatter, _kontext_bauen, _markdown, _namen_farben,
                  _rangfolge, _romanisieren, _sprecher_bei, _zeit,
                  _zusammenfuegen, glossar_lesen, glossar_schreiben,
                  pfad_erlaubt)

HF_TOKEN = os.environ["HF_TOKEN"]
APP_KEY = os.environ["APP_PASS"]
DATEN_REPO = os.environ.get("DATEN_REPO", "Monstermango/diarization-results")
FRONTEND = os.environ.get("FRONTEND_URL",
                          "https://monstermango-sprecher-analyse.static.hf.space")
ZEITZONE = ZoneInfo("Europe/Berlin")

api = HfApi(token=HF_TOKEN)
api.create_repo(DATEN_REPO, repo_type="dataset", private=True, exist_ok=True)

diarizer = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token=HF_TOKEN
)
diarizer.to(torch.device("cuda"))

# Auf Deutsch nachtrainiertes large-v3: bei deutschen Gesprächsaufnahmen
# deutlich treffsicherer als das allgemeine Modell. Fällt bei anderen
# Sprachen auf das Original zurück.
ASR_MODELLE = {
    "german": "primeline/whisper-large-v3-german",
    "*": "openai/whisper-large-v3",
}
_asr_cache = {}


def _asr_holen(sprache):
    name = ASR_MODELLE.get(sprache or "", ASR_MODELLE["*"])
    if name not in _asr_cache:
        _asr_cache[name] = hf_pipeline(
            "automatic-speech-recognition", model=name,
            torch_dtype=torch.float16, device="cuda",
        )
    return _asr_cache[name]


asr = _asr_holen("german")

loader = Audio(sample_rate=16000, mono="downmix")

# Modell zum Einpassen eines mitgelieferten Transkripts
FA_GERAET = torch.device("cuda")
FA_DICT = FA_BUNDLE.get_dict(star="*")
FA_STERN = FA_DICT["*"]
fa_modell = FA_BUNDLE.get_model(with_star=True).to(FA_GERAET).eval()


_fehlversuche = {"anzahl": 0, "zeit": 0.0}


def _pruefe(key):
    """Vergleicht laufzeitkonstant und bremst nach Fehlversuchen.

    Der Space ist öffentlich erreichbar; ohne Bremse ließe sich der
    Schlüssel automatisiert durchprobieren.
    """
    gueltig = bool(key) and hmac.compare_digest(str(key), APP_KEY)
    if gueltig:
        _fehlversuche["anzahl"] = 0
        return True
    jetzt = time.time()
    if jetzt - _fehlversuche["zeit"] > 300:
        _fehlversuche["anzahl"] = 0
    _fehlversuche["anzahl"] += 1
    _fehlversuche["zeit"] = jetzt
    time.sleep(min(5.0, 0.25 * _fehlversuche["anzahl"]))
    return False


# Lange Aufnahmen werden in Fenster zerlegt, damit jeder GPU-Aufruf innerhalb
# des ZeroGPU-Limits bleibt. Die Diarization läuft trotzdem über die gesamte
# Datei, sonst wären die Sprecher zwischen den Fenstern nicht dieselben.


def _diar_dauer(audio_path, num_speakers):
    try:
        laenge = loader.get_duration(audio_path)
    except Exception:
        laenge = 1800
    return int(min(120, max(30, 20 + laenge / 40)))


@spaces.GPU(duration=_diar_dauer)
def _diarize_gpu(audio_path, num_speakers):
    kwargs = {}
    if num_speakers and int(num_speakers) > 0:
        kwargs["num_speakers"] = int(num_speakers)
    output = diarizer(audio_path, **kwargs)
    dia = getattr(output, "speaker_diarization", output)

    turns = [{"start": round(t.start, 2), "ende": round(t.end, 2), "label": lb}
             for t, _, lb in dia.itertracks(yield_label=True)]
    stats = {lb: {"dauer": round(dia.label_duration(lb), 1),
                  "turns": len(dia.label_timeline(lb))}
             for lb in sorted(dia.labels())}
    overlaps = [{"start": round(seg.start, 2), "ende": round(seg.end, 2),
                 "wer": sorted(lb for lb in dia.labels()
                               if dia.label_timeline(lb).crop(seg))}
                for seg in dia.get_overlap()]
    return turns, stats, overlaps


@spaces.GPU(duration=30)
def _sprache_gpu(audio_path, ab):
    """Erkennt die Sprache einmalig anhand von 30 s ab der ersten Sprechstelle."""
    wellenform, sr = loader.crop(audio_path, Segment(ab, ab + 30.0), mode="pad")
    eingabe = {"array": wellenform.squeeze(0).numpy(), "sampling_rate": sr}
    try:
        res = asr(eingabe, return_timestamps=True, return_language=True)
        for c in res.get("chunks", []):
            if c.get("language"):
                return c["language"]
    except (TypeError, ValueError):
        pass
    return None


def _worte_einpassen(audio_path, text, a, b, gesamt):
    """Zerlegt einen erkannten Abschnitt in einzeln verortete Wörter.

    Text und Audio des Abschnitts entsprechen einander exakt, deshalb ist
    das Alignment hier besonders zuverlässig. Nur so lässt sich der Text
    später wortweise dem richtigen Sprecher zuordnen — satzweise Zuordnung
    verteilt Wortwechsel systematisch falsch.
    """
    worte = text.split()
    if not worte:
        return []
    von, bis = max(0.0, a - 0.2), min(gesamt, b + 0.4)
    treffer = []
    if bis - von > 0.1:
        try:
            treffer = _align_fenster(audio_path, worte, von, bis)
        except Exception:
            treffer = []
    if len(treffer) < len(worte) * 0.6:
        # Notfalls gleichmäßig verteilen, damit kein Text verloren geht
        schritt = (b - a) / max(len(worte), 1)
        return [{"start": a + i * schritt, "ende": a + (i + 1) * schritt,
                 "text": w, "sprache": None} for i, w in enumerate(worte)]
    return [{"start": s, "ende": e, "text": worte[nr], "sprache": None}
            for nr, s, e in treffer]


def _asr_dauer(audio_path, start, ende, sprache=None, kontext=""):
    # Grosszuegiger als die reine Rechenzeit: der Temperatur-Rückfall
    # verwirft entgleiste Abschnitte und dekodiert sie erneut, im
    # schlimmsten Fall fünfmal. Das Budget ist eine Reservierung, keine
    # Abrechnung — zu knapp bemessen bricht der Lauf mittendrin ab.
    return int(min(120, max(60, 40 + (ende - start) / 2)))


@spaces.GPU(duration=_asr_dauer)
def _transkribiere_gpu(audio_path, start, ende, sprache=None, kontext=""):
    """Transkribiert ein Fenster am Stück — voller Kontext, beste Qualität.

    Die Sprache wird fest vorgegeben: sonst rät Whisper sie pro Abschnitt
    neu und kippt dabei gern mitten im Gespräch in eine englische
    Übersetzung.
    """
    modell = _asr_holen(sprache)
    wellenform, sr = loader.crop(audio_path, Segment(start, ende), mode="pad")
    eingabe = {"array": wellenform.squeeze(0).numpy(), "sampling_rate": sr}
    # Bewusst ohne prompt_ids — ein Vorwissen-Prompt bringt die
    # Langform-Dekodierung zum Verstummen.
    gk = _decode_optionen(sprache)
    try:
        res = modell(eingabe, return_timestamps=True, generate_kwargs=gk,
                     return_language=not sprache)
    except (TypeError, ValueError):
        res = modell(eingabe, return_timestamps=True, generate_kwargs=gk)

    sprache_erkannt = None
    worte = []
    for c in res.get("chunks", []):
        ts = c.get("timestamp") or (None, None)
        text = (c.get("text") or "").strip()
        if ts[0] is None or not text:
            continue
        sprache_erkannt = sprache_erkannt or c.get("language")
        bis = ts[1] if ts[1] is not None else ts[0] + 30.0
        worte += _worte_einpassen(audio_path, text, float(ts[0]) + start,
                                  float(bis) + start, ende)
    if not worte:
        # Manche Modelle (etwa deutsche Feinabstimmungen) liefern keine
        # Zeitmarken. Dann den gesamten Text des Fensters selbst verorten.
        alle = (res.get("text") or "").strip().split()
        if alle:
            rate = len(alle) / max(ende - start, 1.0)
            treffer, _, _ = _aligniere_bereich(audio_path, alle, start, ende,
                                               rate, schluss=True)
            worte = [{"start": a, "ende": b, "text": alle[nr], "sprache": None}
                     for nr, a, b in treffer]

    for w in worte:
        w["sprache"] = sprache_erkannt or sprache
    return worte


# ---------- Eigenes Transkript einpassen (Forced Alignment) ----------
# Wird ein fertiges Transkript mitgeliefert (z. B. aus Apples Sprachmemos),
# muss nicht neu erkannt werden: der Text wird direkt auf die Tonspur gelegt.
# Das ist schneller als Whisper und übernimmt dessen bessere Textqualität.

def _align_fenster(audio_path, worte, start, ende):
    """Ordnet Wörter einem Audiofenster zu -> [(index, start_s, ende_s)].

    Es wird bewusst weniger Text übergeben, als das Fenster fassen kann;
    das überschüssige Audio fängt der Stern-Token am Ende auf. Andernfalls
    müsste der Aligner die Wörter zusammenstauchen und die Zeiten wären
    unbrauchbar.
    """
    tokens, idx = [], []
    for i, w in enumerate(worte):
        t = [FA_DICT[c] for c in _romanisieren(w) if c in FA_DICT]
        if t:
            tokens.append(t)
            idx.append(i)
    if not tokens:
        return []

    wellenform, sr = loader.crop(audio_path, Segment(start, ende), mode="pad")
    welle = wellenform.to(FA_GERAET)
    with torch.inference_mode():
        emission, _ = fa_modell(welle)
    # Stern nur am Ende: das Fenster beginnt exakt dort, wo die zuletzt
    # zugeordneten Wörter endeten — ein führender Stern würde den Aligner
    # verleiten, den Anfang zu überspringen.
    ziel = torch.tensor([[t for wort in tokens for t in wort] + [FA_STERN]],
                        dtype=torch.int32, device=FA_GERAET)
    pfad, punkte = AF.forced_align(emission, ziel, blank=0)
    spans = [s for s in AF.merge_tokens(pfad[0], punkte[0]) if s.token != FA_STERN]

    verhaeltnis = welle.size(1) / emission.size(1) / sr
    ergebnis, pos = [], 0
    for wort_nr, tok in zip(idx, tokens):
        teil = spans[pos:pos + len(tok)]
        pos += len(tok)
        if teil:
            ergebnis.append((wort_nr, start + teil[0].start * verhaeltnis,
                             start + teil[-1].end * verhaeltnis))
    return ergebnis


def _aligniere_bereich(audio_path, worte, start, ende, rate, schluss):
    """Verteilt eine Wortliste über einen Zeitbereich, Fenster für Fenster.

    `schluss` gibt an, ob das Bereichsende zugleich das Ende des Materials
    ist — nur dann muss der verbleibende Text vollständig untergebracht
    werden.
    """
    treffer_gesamt, zeit, wi = [], start, 0
    while wi < len(worte) and zeit < ende - 0.5:
        fenster_ende = min(ende, zeit + FA_FENSTER)
        if schluss and fenster_ende >= ende - 0.5:
            nehmen = len(worte) - wi          # Rest muss vollständig hinein
        else:
            nehmen = max(3, int((fenster_ende - zeit) * rate * FA_ANTEIL))
        treffer = _align_fenster(audio_path, worte[wi:wi + nehmen],
                                 zeit, fenster_ende)
        if not treffer:
            zeit = fenster_ende
            continue
        for nr, a, b in treffer:
            treffer_gesamt.append((wi + nr, a, b))
        wi += treffer[-1][0] + 1
        zeit = max(treffer[-1][2], zeit + 1.0)
    return treffer_gesamt, zeit, wi


def _fa_dauer(audio_path, worte, start, block_ende, rate, gesamt):
    return int(min(120, max(30, 20 + (block_ende - start) / 10)))


@spaces.GPU(duration=_fa_dauer)
def _align_gpu(audio_path, worte, start, block_ende, rate, gesamt):
    """Passt Wörter fortlaufend in einen Audioblock ein.

    Mehrere Fenster je GPU-Aufruf, damit lange Aufnahmen nicht in sehr
    viele Einzelanfragen zerfallen.
    """
    return _aligniere_bereich(audio_path, worte, start, block_ende, rate,
                              schluss=block_ende >= gesamt - 0.5)


def _align_schritt(s):
    """Passt den nächsten Audioblock des Transkripts ein. -> weiter?"""
    worte, gesamt = s["worte"], s["dauer"]
    if s["fa_wort"] >= len(worte) or s["fa_zeit"] >= gesamt - 0.5:
        return False

    block_ende = min(gesamt, s["fa_zeit"] + FA_BLOCK)
    rate = len(worte) / max(gesamt, 1.0)
    treffer, neue_zeit, verbraucht = _align_gpu(
        s["wav"], worte[s["fa_wort"]:], s["fa_zeit"], block_ende, rate, gesamt)

    for nr, a, b in treffer:
        s["chunks"].append({"start": a, "ende": b,
                            "text": worte[s["fa_wort"] + nr], "sprache": None})
    s["fa_wort"] += verbraucht
    s["fa_zeit"] = max(neue_zeit, s["fa_zeit"] + 1.0)
    return s["fa_wort"] < len(worte) and s["fa_zeit"] < gesamt - 0.5


# ---------- Sitzungen ----------
# Lange Aufnahmen werden über mehrere HTTP-Anfragen verarbeitet: ein einzelner
# Aufruf würde länger dauern als das ZeroGPU-Token einer Anfrage gültig ist.

SITZUNGEN = {}
SITZUNG_TTL = 3 * 3600


def _dateien_loeschen(s):
    """Entfernt Audio vom Server — Originalupload wie umgewandelte Fassung.

    Gesprächsaufnahmen sind das Sensibelste, was durch dieses System läuft;
    sie dürfen nicht länger liegen bleiben als für die Analyse nötig.
    """
    for schluessel in ("wav", "original"):
        pfad = s.get(schluessel)
        if not pfad:
            continue
        try:
            os.unlink(pfad)
        except OSError:
            pass
        s[schluessel] = None


def _aufraeumen():
    jetzt = time.time()
    for sid in [s for s, v in SITZUNGEN.items()
                if jetzt - v["zeit"] > SITZUNG_TTL]:
        _dateien_loeschen(SITZUNGEN[sid])
        SITZUNGEN.pop(sid, None)


def _aufraeum_schleife():
    """Räumt auch dann auf, wenn keine neue Aufnahme mehr hochgeladen wird."""
    while True:
        time.sleep(600)
        try:
            _aufraeumen()
        except Exception:
            traceback.print_exc()


threading.Thread(target=_aufraeum_schleife, daemon=True).start()


def _sitzung(sid):
    s = SITZUNGEN.get(sid)
    if s:
        s["zeit"] = time.time()
    return s


# ---------- Markdown-Ablage ----------

def _speichern(md_text, zeitpunkt):
    pfad = f"aufnahmen/{zeitpunkt:%Y-%m-%d_%H%M%S}.md"
    api.upload_file(path_or_fileobj=md_text.encode("utf-8"), path_in_repo=pfad,
                    repo_id=DATEN_REPO, repo_type="dataset",
                    commit_message=f"Analyse {pfad}")
    return pfad


# ---------- Glossar ----------
# Wiederkehrende Begriffe und Sprechernamen liegen im selben privaten
# Dataset. Sie werden bei jeder Analyse als Vorwissen an die Erkennung
# gegeben, damit Eigennamen und Fachbegriffe über Aufnahmen hinweg besser
# getroffen werden.

_glossar_cache = {"zeit": 0.0, "daten": None}
GLOSSAR_TTL = 60.0

# Häufige groß geschriebene Wörter, die als Vorschlag nichts bringen


def _glossar_laden(frisch=False):
    """Begriffe mit Häufigkeit, absteigend sortiert."""
    jetzt = time.time()
    if (not frisch and _glossar_cache["daten"] is not None
            and jetzt - _glossar_cache["zeit"] < GLOSSAR_TTL):
        return _glossar_cache["daten"]
    zaehler, sprecher = {}, []
    try:
        pfad = hf_hub_download(DATEN_REPO, GLOSSAR_DATEI, repo_type="dataset",
                               token=HF_TOKEN, force_download=frisch)
        with open(pfad, encoding="utf-8") as fh:
            zaehler, sprecher = glossar_lesen(fh.read())
    except Exception:
        pass
    daten = {"zaehler": zaehler, "sprecher": sprecher,
             "begriffe": _rangfolge(zaehler)}
    _glossar_cache.update(zeit=jetzt, daten=daten)
    return daten


def _glossar_speichern(zaehler, sprecher):
    text, geordnet = glossar_schreiben(zaehler, sprecher)
    api.upload_file(path_or_fileobj=text.encode("utf-8"),
                    path_in_repo=GLOSSAR_DATEI, repo_id=DATEN_REPO,
                    repo_type="dataset", commit_message="Glossar aktualisiert")
    behalten = {b: zaehler[b] for b in geordnet}
    _glossar_cache.update(zeit=time.time(),
                          daten={"zaehler": behalten, "sprecher": sprecher,
                                 "begriffe": geordnet})


def _glossar_fortschreiben(segmente, sprechernamen):
    """Zählt Begriffe des neuen Transkripts mit — ohne Zutun des Nutzers.

    Gibt zusätzlich zurück, was neu hinzugekommen ist: nur so kann die App
    nachvollziehbar machen, dass und was das Glossar gelernt hat.
    """
    try:
        g = _glossar_laden(frisch=True)
        zaehler = dict(g["zaehler"])
        neu = []
        for wort, anzahl in _begriffe_zaehlen(segmente).items():
            if wort not in zaehler:
                neu.append(wort)
            zaehler[wort] = zaehler.get(wort, 0) + anzahl
        sprecher = list(g["sprecher"])
        for name in sprechernamen:
            if name and name not in sprecher:
                sprecher.append(name)
                neu.append(name)
        _glossar_speichern(zaehler, sprecher)
        return _glossar_laden(), neu
    except Exception:
        traceback.print_exc()
        return _glossar_laden(), []


def glossar_api(key):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    g = _glossar_laden(frisch=True)
    return {"ok": True, **g}


def glossar_speichern_api(key, begriffe_text, sprecher_text):
    """Manuelles Nachbearbeiten; die Reihenfolge legt den Rang fest."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}

    def zerlegen(t):
        raus = []
        for w in re.split(r"[\n,;]+", t or ""):
            w = re.sub(r"\s*\(\d+\)$", "", w.strip(" -\t")).strip()
            if w and w not in raus:
                raus.append(w)
        return raus[:GLOSSAR_MAX]

    begriffe, sprecher = zerlegen(begriffe_text), zerlegen(sprecher_text)
    alt = _glossar_laden(frisch=True)["zaehler"]
    # Vorhandene Häufigkeiten behalten, neue oben einsortieren
    hoechster = max(alt.values(), default=1)
    zaehler = {}
    for rang, b in enumerate(begriffe):
        zaehler[b] = alt.get(b, max(2, hoechster - rang))
    try:
        _glossar_speichern(zaehler, sprecher)
    except Exception as e:
        return {"ok": False, "fehler": f"Speichern fehlgeschlagen: {e}"}
    g = _glossar_laden()
    return {"ok": True, "begriffe": g["begriffe"], "sprecher": g["sprecher"],
            "zaehler": g["zaehler"]}


# ---------- API-Endpunkte ----------

def _nach_wav(pfad):
    """Konvertiert nach 16-kHz-Mono-WAV und gleicht die Lautstärke an.

    Aufnahmen mit Abstand zum Mikrofon sind oft sehr leise und schwanken
    stark; die Pegelangleichung verbessert Sprechererkennung und
    Transkription spürbar.
    """
    ziel = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    filter_kette = "highpass=f=60,loudnorm=I=-18:LRA=11:TP=-2,dynaudnorm=f=200:g=5"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pfad,
                        "-af", filter_kette, "-ar", "16000", "-ac", "1", ziel],
                       check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pfad,
                        "-ar", "16000", "-ac", "1", ziel], check=True)
    return ziel


def status_api(key, request: gr.Request = None):
    """Meldet, ob die Anfrage mit HF-Token ankommt (GPU-Kontingent-Zuordnung)."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    kopf = {}
    try:
        kopf = dict(request.headers) if request is not None else {}
    except Exception:
        pass
    auth = kopf.get("authorization") or kopf.get("Authorization") or ""
    return {"ok": True, "token": auth.lower().startswith("bearer hf_")}


def vorbereiten_api(key, audio, transkript=None, kontext=None):
    """Schritt 1: Audio entgegennehmen und in WAV wandeln.

    Wird ein Transkript mitgegeben, wird dieses später eingepasst statt
    selbst zu erkennen.
    """
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    if not audio:
        return {"ok": False, "fehler": "Kein Audio übermittelt."}
    _aufraeumen()
    quelle = os.path.basename(audio)
    try:
        wav = _nach_wav(audio)
    except subprocess.CalledProcessError:
        return {"ok": False, "fehler": "Audioformat konnte nicht gelesen werden."}
    worte = (transkript or "").split()
    sid = secrets.token_urlsafe(16)
    SITZUNGEN[sid] = {"wav": wav, "original": audio,
                      "quelle": quelle, "zeit": time.time(),
                      "dauer": loader.get_duration(wav), "chunks": [],
                      "worte": worte, "fa_zeit": 0.0, "fa_wort": 0,
                      "eigener": (kontext or "").strip(),
                      "kontext": _kontext_bauen((kontext or "").strip(),
                                                _glossar_laden())}
    return {"ok": True, "id": sid, "dauer": round(SITZUNGEN[sid]["dauer"], 1),
            "modus": "transkript" if worte else "erkennung",
            "woerter": len(worte)}


def diarisieren_api(key, sid, num_speakers, sprache=None):
    """Schritt 2: Sprecher über die gesamte Aufnahme erkennen."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    s = _sitzung(sid)
    if not s:
        return {"ok": False, "fehler": "Sitzung abgelaufen. Bitte erneut starten."}
    try:
        turns, stats, overlaps = _diarize_gpu(s["wav"], num_speakers)
    except Exception as e:
        traceback.print_exc()
        return {"ok": False,
                "fehler": f"Sprechererkennung fehlgeschlagen: {type(e).__name__}: {e}"}
    sprache = (sprache or "").strip() or None
    if not sprache and not s.get("worte"):
        try:
            sprache = _sprache_gpu(s["wav"], turns[0]["start"] if turns else 0.0)
        except Exception:
            sprache = None
    s.update(turns=turns, stats=stats, overlaps=overlaps, chunks=[],
             sprache=sprache, fa_zeit=0.0, fa_wort=0, fertig=set(),
             fenster=_fenstergrenzen(turns, s["dauer"]))
    abschnitte = (max(1, math.ceil(s["dauer"] / FA_FENSTER)) if s.get("worte")
                  else max(1, len(s["fenster"]) - 1))
    return {"ok": True, "sprecher": len(stats), "sprache": sprache,
            "abschnitte": abschnitte}


def transkribieren_api(key, sid, index):
    """Schritt 3: nächsten Abschnitt verarbeiten (eigene Anfrage je Abschnitt)."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    s = _sitzung(sid)
    if not s or "fenster" not in s:
        return {"ok": False, "fehler": "Sitzung abgelaufen. Bitte erneut starten."}

    if s.get("worte"):
        try:
            weiter = _align_schritt(s)
        except Exception as e:
            traceback.print_exc()
            return {"ok": False,
                    "fehler": f"Transkript einpassen fehlgeschlagen: "
                              f"{type(e).__name__}: {e}"}
        return {"ok": True, "weiter": weiter,
                "fortschritt": round(min(1.0, s["fa_zeit"] / max(s["dauer"], 1)), 2)}

    i = int(index)
    if not 0 <= i < len(s["fenster"]) - 1:
        return {"ok": False, "fehler": "Ungültiger Abschnitt."}
    if i in s.setdefault("fertig", set()):
        # Bereits verarbeitet — Wiederholung nach Abbruch darf nichts doppeln
        return {"ok": True, "index": i, "abschnitte": len(s["fenster"]) - 1,
                "weiter": i + 1 < len(s["fenster"]) - 1}
    try:
        s["chunks"] += _transkribiere_gpu(s["wav"], s["fenster"][i],
                                          s["fenster"][i + 1],
                                          s.get("sprache"), s.get("kontext", ""))
    except Exception as e:
        traceback.print_exc()
        return {"ok": False,
                "fehler": f"Transkription fehlgeschlagen: {type(e).__name__}: {e}"}
    s["fertig"].add(i)
    return {"ok": True, "index": i, "abschnitte": len(s["fenster"]) - 1,
            "weiter": i + 1 < len(s["fenster"]) - 1}


def _daten_sichern(s):
    if "daten" not in s:
        daten = _zusammenfuegen(s["turns"], s["stats"], s["overlaps"],
                                s["chunks"], s["dauer"], s.get("sprache"))
        # Erst nach der Zuordnung korrigieren: die Zeitstempel stammen aus
        # dem Audio und dürfen von einer Textänderung nicht berührt werden.
        # Ein mitgeliefertes Transkript bleibt unangetastet — es ist die
        # Vorlage, nicht die Vermutung.
        if not s.get("worte"):
            ziele = _korrekturziele(_glossar_laden(), s.get("eigener", ""))
            daten["segmente"], daten["korrekturen"] = _korrigieren(
                daten["segmente"], ziele)
        else:
            daten["korrekturen"] = []
        s["daten"] = daten
    return s["daten"]


def abschliessen_api(key, sid):
    """Schritt 5: Ergebnis ausliefern und das Glossar fortschreiben."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    s = _sitzung(sid)
    if not s or "turns" not in s:
        return {"ok": False, "fehler": "Sitzung abgelaufen. Bitte erneut starten."}
    daten = _daten_sichern(s)
    # Ab hier wird das Audio nicht mehr gebraucht — sofort entfernen.
    _dateien_loeschen(s)
    zeitpunkt = datetime.now(ZEITZONE)
    namen, farben = _namen_farben(daten["stats"])
    glossar, neue_begriffe = _glossar_fortschreiben(daten["segmente"], [])
    return {"ok": True, "daten": daten, "namen": namen, "farben": farben,
            "zeitpunkt": zeitpunkt.isoformat(timespec="seconds"),
            "quelle": s["quelle"], "sprache": daten.get("sprache"),
            "glossar": glossar["begriffe"][:GLOSSAR_PROMPT],
            "glossar_gesamt": len(glossar["begriffe"]),
            "neue_begriffe": neue_begriffe,
            "sprecher_bekannt": glossar["sprecher"]}


def analysieren_api(key, audio, num_speakers, sprache=None, transkript=None,
                    kontext=None):
    """Alle Schritte in einem Aufruf — nur für kurze Aufnahmen geeignet."""
    vor = vorbereiten_api(key, audio, transkript, kontext)
    if not vor.get("ok"):
        return vor
    sid = vor["id"]
    dia = diarisieren_api(key, sid, num_speakers, sprache)
    if not dia.get("ok"):
        return dia
    i = 0
    while True:
        tr = transkribieren_api(key, sid, i)
        if not tr.get("ok"):
            return tr
        if not tr.get("weiter"):
            break
        i += 1
    return abschliessen_api(key, sid)


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

    neue = []
    if eigene:                      # Sprechernamen fürs nächste Mal merken
        _, neue = _glossar_fortschreiben([], list(eigene.values()))
    return {"ok": True, "pfad": pfad, "markdown": md_text,
            "neue_begriffe": neue}


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
    if not pfad_erlaubt(pfad):
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
    gr.Button("Status prüfen").click(status_api, [key], gr.JSON(),
                                     api_name="status")
    with gr.Accordion("Glossar", open=False):
        g_begriffe = gr.Textbox(label="Begriffe (je Zeile einer)", lines=4)
        g_sprecher = gr.Textbox(label="Sprechernamen (je Zeile einer)", lines=3)
        gr.Button("Glossar laden").click(glossar_api, [key], gr.JSON(),
                                         api_name="glossar")
        gr.Button("Glossar speichern").click(
            glossar_speichern_api, [key, g_begriffe, g_sprecher], gr.JSON(),
            api_name="glossar_speichern")
    with gr.Tab("Analyse"):
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                         label="Audio")
        num_speakers = gr.Number(value=0, precision=0,
                                 label="Anzahl Sprecher (0 = automatisch)")
        sprache_feld = gr.Textbox(label="Sprache (leer = automatisch)",
                                  value="german")
        transkript_feld = gr.Textbox(
            label="Eigenes Transkript (leer = selbst erkennen)", lines=4)
        kontext_feld = gr.Textbox(label="Namen & Begriffe (optional)")
        b_analyse = gr.Button("Analysieren", variant="primary")
        out_analyse = gr.JSON()
        b_analyse.click(analysieren_api,
                        [key, audio, num_speakers, sprache_feld, transkript_feld,
                         kontext_feld],
                        out_analyse, api_name="analysieren")
    with gr.Tab("Schrittweise (lange Aufnahmen)"):
        sid = gr.Textbox(label="Sitzungs-ID")
        idx = gr.Number(value=0, precision=0, label="Abschnitt")
        gr.Button("1 · Vorbereiten").click(
            vorbereiten_api, [key, audio, transkript_feld, kontext_feld],
            gr.JSON(), api_name="vorbereiten")
        gr.Button("2 · Sprecher erkennen").click(
            diarisieren_api, [key, sid, num_speakers, sprache_feld], gr.JSON(),
            api_name="diarisieren")
        gr.Button("3 · Abschnitt transkribieren").click(
            transkribieren_api, [key, sid, idx], gr.JSON(),
            api_name="transkribieren")
        gr.Button("4 · Abschließen").click(
            abschliessen_api, [key, sid], gr.JSON(), api_name="abschliessen")
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
