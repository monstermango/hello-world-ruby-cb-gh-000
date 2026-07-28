"""API-Backend der Sprecher-Analyse (HF Space auf ZeroGPU).

Die eigentliche App (PWA) läuft als eigener Static Space und spricht die
hier registrierten API-Endpunkte an. Jeder Endpunkt verlangt den
Zugangsschlüssel (Secret APP_PASS); erst nach der Prüfung wird GPU-Zeit
verbraucht.
"""
import contextvars
import hmac
import json
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
                  _ist_halluzination, _konfidenz_markieren, _korrekturziele,
                  _korrigieren, abgleich, aufnahme_urteil,
                  KORREKTUR_MIN_ZAEHLER, stimmen_zuordnen,
                  stimmen_auffrischen,
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

    # Der Stimmabdruck fällt beim Clustern ohnehin an. Aufgehoben erkennt
    # die App dieselbe Person beim nächsten Mal wieder, statt sie erneut
    # als „Sprecher 2“ zu begrüßen. Nicht jede Fassung gibt ihn heraus,
    # deshalb der Rückfall auf den gewöhnlichen Aufruf.
    einbettungen, output = None, None
    try:
        output, einbettungen = diarizer(audio_path, return_embeddings=True,
                                        **kwargs)
    except (TypeError, ValueError):
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

    abdruecke = {}
    if einbettungen is not None:
        try:
            for lb, vektor in zip(sorted(dia.labels()), einbettungen):
                werte = [float(x) for x in vektor]
                if any(werte):
                    abdruecke[lb] = werte
        except (TypeError, ValueError):
            abdruecke = {}
    return turns, stats, overlaps, abdruecke


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
        # Gleichmäßig verteilt heißt: nicht wirklich verortet. Das ist
        # genau der Fall, in dem man dem Wort nicht trauen sollte.
        return [{"start": a + i * schritt, "ende": a + (i + 1) * schritt,
                 "text": w, "sprache": None, "wert": 0.0}
                for i, w in enumerate(worte)]
    return [{"start": s, "ende": e, "text": worte[nr], "sprache": None,
             "wert": wert} for nr, s, e, wert in treffer]


def _pegel_messen(wav_pfad, turns=None):
    """Misst Sprachpegel, Rauschteppich und Übersteuerung.

    Was nicht im Signal steckt, holt kein Modell zurück. Diese Grenze war
    bisher unsichtbar: der Nutzer sah nur ein schlechtes Ergebnis und
    konnte nicht wissen, dass es an der Aufnahme lag. Liegen bereits
    Sprecherzeiten vor, wird Sprache gegen Nichtsprache gemessen — sonst
    ersatzweise das laute gegen das leise Zehntel.
    """
    import numpy as np
    try:
        welle, sr = loader({"audio": wav_pfad})
        x = welle.squeeze(0).numpy().astype("float32")
    except Exception:
        return None
    if x.size < 1000:
        return None

    fenster = max(1, int(sr * 0.05))
    n = x.size // fenster
    if n < 4:
        return None
    rahmen = x[:n * fenster].reshape(n, fenster)
    energie = np.sqrt((rahmen ** 2).mean(axis=1)) + 1e-9
    db = 20.0 * np.log10(energie)

    if turns:
        ist_sprache = np.zeros(n, dtype=bool)
        for t in turns:
            a = max(0, int(t["start"] * sr / fenster))
            b = min(n, int(t["ende"] * sr / fenster) + 1)
            ist_sprache[a:b] = True
        if ist_sprache.any() and (~ist_sprache).any():
            sprache_db = float(np.percentile(db[ist_sprache], 75))
            rausch_db = float(np.percentile(db[~ist_sprache], 50))
        else:
            turns = None
    if not turns:
        sprache_db = float(np.percentile(db, 90))
        rausch_db = float(np.percentile(db, 10))

    return {"sprache_db": sprache_db, "rausch_db": rausch_db,
            "uebersteuert": float((np.abs(x) > 0.995).mean())}


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
        if ts[0] is None or not text or _ist_halluzination(text):
            continue
        sprache_erkannt = sprache_erkannt or c.get("language")
        bis = ts[1] if ts[1] is not None else ts[0] + 30.0
        worte += _worte_einpassen(audio_path, text, float(ts[0]) + start,
                                  float(bis) + start, ende)
    if not worte:
        # Manche Modelle (etwa deutsche Feinabstimmungen) liefern keine
        # Zeitmarken. Dann den gesamten Text des Fensters selbst verorten.
        ganz = (res.get("text") or "").strip()
        alle = [] if _ist_halluzination(ganz) else ganz.split()
        if alle:
            rate = len(alle) / max(ende - start, 1.0)
            treffer, _, _ = _aligniere_bereich(audio_path, alle, start, ende,
                                               rate, schluss=True)
            worte = [{"start": a, "ende": b, "text": alle[nr],
                      "sprache": None, "wert": wert}
                     for nr, a, b, wert in treffer]

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
            # Der Anpassungswert sagt, wie gut das Wort im Audio
            # wiedergefunden wurde. Bisher weggeworfen — damit sah ein
            # wackliges Wort so sicher aus wie ein zweifelsfreies.
            wert = sum(sp.score for sp in teil) / len(teil)
            ergebnis.append((wort_nr, start + teil[0].start * verhaeltnis,
                             start + teil[-1].end * verhaeltnis, float(wert)))
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
        for nr, a, b, wert in treffer:
            treffer_gesamt.append((wi + nr, a, b, wert))
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

    for nr, a, b, wert in treffer:
        s["chunks"].append({"start": a, "ende": b, "wert": wert,
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


STIMMEN_DATEI = "stimmen.json"
_stimmen_cache = {"zeit": 0.0, "daten": None}


def _stimmen_laden(frisch=False):
    """Gespeicherte Stimmabdrücke je benanntem Sprecher."""
    jetzt = time.time()
    if (not frisch and _stimmen_cache["daten"] is not None
            and jetzt - _stimmen_cache["zeit"] < GLOSSAR_TTL):
        return _stimmen_cache["daten"]
    daten = {}
    try:
        pfad = hf_hub_download(DATEN_REPO, STIMMEN_DATEI, repo_type="dataset",
                               token=HF_TOKEN, force_download=frisch)
        with open(pfad, encoding="utf-8") as fh:
            daten = json.load(fh)
    except Exception:
        pass
    _stimmen_cache.update(zeit=jetzt, daten=daten)
    return daten


def _stimmen_sichern(daten):
    api.upload_file(path_or_fileobj=json.dumps(daten).encode("utf-8"),
                    path_in_repo=STIMMEN_DATEI, repo_id=DATEN_REPO,
                    repo_type="dataset",
                    commit_message="Stimmabdrücke aktualisiert")
    _stimmen_cache.update(zeit=time.time(), daten=daten)


def _glossar_laden(frisch=False):
    """Begriffe mit Häufigkeit, absteigend sortiert."""
    jetzt = time.time()
    if (not frisch and _glossar_cache["daten"] is not None
            and jetzt - _glossar_cache["zeit"] < GLOSSAR_TTL):
        return _glossar_cache["daten"]
    zaehler, sprecher, festen = {}, [], {}
    try:
        pfad = hf_hub_download(DATEN_REPO, GLOSSAR_DATEI, repo_type="dataset",
                               token=HF_TOKEN, force_download=frisch)
        with open(pfad, encoding="utf-8") as fh:
            zaehler, sprecher, festen = glossar_lesen(fh.read())
    except Exception:
        pass
    daten = {"zaehler": zaehler, "sprecher": sprecher, "festen": festen,
             "begriffe": _rangfolge(zaehler)}
    _glossar_cache.update(zeit=jetzt, daten=daten)
    return daten


def _glossar_speichern(zaehler, sprecher, festen=None):
    text, geordnet = glossar_schreiben(zaehler, sprecher, festen)
    api.upload_file(path_or_fileobj=text.encode("utf-8"),
                    path_in_repo=GLOSSAR_DATEI, repo_id=DATEN_REPO,
                    repo_type="dataset", commit_message="Glossar aktualisiert")
    behalten = {b: zaehler[b] for b in geordnet}
    _glossar_cache.update(zeit=time.time(),
                          daten={"zaehler": behalten, "sprecher": sprecher,
                                 "festen": festen or {},
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
        _glossar_speichern(zaehler, sprecher, g.get("festen"))
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
    vorher = _glossar_laden(frisch=True)
    alt = vorher["zaehler"]
    # Vorhandene Häufigkeiten behalten, neue oben einsortieren
    hoechster = max(alt.values(), default=1)
    zaehler = {}
    for rang, b in enumerate(begriffe):
        zaehler[b] = alt.get(b, max(2, hoechster - rang))
    try:
        _glossar_speichern(zaehler, sprecher, vorher.get("festen"))
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
    # Ohne dynaudnorm. Die dynamische Normalisierung zieht leise Stellen
    # hoch — in Sprechpausen also den Rauschteppich, und auf angehobenes
    # Rauschen antwortet Whisper mit erfundenem Text. Sie hebelt zugleich
    # no_speech_threshold aus, das Stille an ihrem Pegel erkennt. Bleiben
    # Hochpass gegen Trittschall und loudnorm für einen gleichmäßigen
    # Pegel; Whisper ist auf unbearbeitetem Alltagsmaterial trainiert und
    # braucht mehr nicht.
    filter_kette = "highpass=f=60,loudnorm=I=-18:LRA=11:TP=-2"
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pfad,
                        "-af", filter_kette, "-ar", "16000", "-ac", "1", ziel],
                       check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pfad,
                        "-ar", "16000", "-ac", "1", ziel], check=True)
    return ziel


# ---------- Hintergrundverarbeitung ----------
# Bisher war der Browser der Taktgeber: er rief Schritt für Schritt auf.
# Schläft das Telefon, steht die Verarbeitung. Der Space kann die Schritte
# aber selbst durchlaufen — nachgemessen laufen sie dabei auf dem
# PRO-Kontingent des Space-Besitzers, nicht auf dem anonymen.

AUFTRAEGE = {}
AUFTRAG_TTL = 6 * 3600


def _auftrag_lauf(jid, sid, num_speakers, sprache):
    """Fährt die komplette Kette durch, ohne dass jemand zusieht."""
    j = AUFTRAEGE[jid]
    try:
        r = diarisieren_api(APP_KEY, sid, num_speakers, sprache)
        if not r.get("ok"):
            raise RuntimeError(r.get("fehler", "Sprechererkennung fehlgeschlagen"))
        gesamt = max(1, int(r.get("abschnitte") or 1))
        j.update(stand="laeuft", schritt="Text erkennen", von=0, bis=gesamt)

        i, getan = 0, 0
        while True:
            a = transkribieren_api(APP_KEY, sid, i)
            if not a.get("ok"):
                raise RuntimeError(a.get("fehler", "Transkription fehlgeschlagen"))
            getan += 1
            j.update(von=min(getan, gesamt))
            if not a.get("weiter"):
                break
            # Beim Einpassen zählt das Backend selbst weiter, bei der
            # Fenster-Erkennung der Index.
            i = a.get("index", i) + 1 if "index" in a else i

        j.update(schritt="Zusammenstellen")
        e = abschliessen_api(APP_KEY, sid)
        if not e.get("ok"):
            raise RuntimeError(e.get("fehler", "Abschluss fehlgeschlagen"))
        j.update(stand="fertig", ergebnis=e, zeit=time.time())
        _push_senden("Analyse fertig",
                     f"{len(e.get('daten', {}).get('stats', {}))} Sprecher, "
                     f"{_zeit(e.get('daten', {}).get('dauer', 0))} min")
    except Exception as ex:
        traceback.print_exc()
        j.update(stand="fehler", fehler=f"{type(ex).__name__}: {ex}",
                 zeit=time.time())
        _push_senden("Analyse fehlgeschlagen", str(ex)[:120])


def starten_api(key, sid, num_speakers, sprache=None, request: gr.Request = None):
    """Stößt die Verarbeitung an und gibt sofort zurück."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    if not _sitzung(sid):
        return {"ok": False, "fehler": "Sitzung abgelaufen. Bitte erneut starten."}

    # Festhalten, womit gestartet wurde. ZeroGPU rechnet die GPU-Zeit dem
    # zu, dessen Kennung an der Anfrage hängt; ob sie anlag, soll nicht
    # Glaubenssache sein, sondern ablesbar.
    kopf = {}
    try:
        kopf = {k.lower(): v for k, v in dict(request.headers).items()}
    except Exception:
        pass
    mit_token = bool(
        (kopf.get("authorization") or "").lower().startswith("bearer hf_")
        or kopf.get("x-ip-token"))

    jid = secrets.token_urlsafe(12)
    AUFTRAEGE[jid] = {"stand": "laeuft", "schritt": "Sprecher erkennen",
                      "von": 0, "bis": 1, "sid": sid, "zeit": time.time(),
                      "mit_token": mit_token}

    # Den Kontext dieser Anfrage mitnehmen. ZeroGPU ordnet die GPU-Zeit
    # anhand eines Kopfzeilenwerts zu, den die Hub-Infrastruktur pro
    # Anfrage setzt — und dieser Aufruf ist die Anfrage des Nutzers, mit
    # seinem Token. Ein Thread erbt Kontextvariablen nicht von allein;
    # ohne das Kopieren fiele die Zuordnung an der Threadgrenze weg.
    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run,
                     args=(_auftrag_lauf, jid, sid, num_speakers, sprache),
                     daemon=True).start()
    return {"ok": True, "auftrag": jid, "mit_token": mit_token}


def auftrag_api(key, jid):
    """Fragt den Stand ab. Der Browser darf zwischendurch schlafen."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    j = AUFTRAEGE.get(jid)
    if not j:
        return {"ok": False, "fehler": "Auftrag unbekannt oder abgelaufen."}
    antwort = {k: v for k, v in j.items() if k != "ergebnis"}
    if j.get("stand") == "fertig":
        antwort["ergebnis"] = j["ergebnis"]
    return {"ok": True, **antwort}


# ---------- Benachrichtigung ----------
# Web Push funktioniert auf dem iPhone nur für eine zum Home-Bildschirm
# hinzugefügte App. Der Inhalt ist Ende-zu-Ende verschlüsselt: Apples
# Zustelldienst sieht nur Chiffrat, keine Gesprächsdaten.

PUSH_DATEI = "push.json"
_push_cache = {"daten": None}


def _push_zustand():
    if _push_cache["daten"] is None:
        daten = {"vapid_pem": None, "abos": []}
        try:
            pfad = hf_hub_download(DATEN_REPO, PUSH_DATEI, repo_type="dataset",
                                   token=HF_TOKEN, force_download=True)
            with open(pfad, encoding="utf-8") as fh:
                daten = json.load(fh)
        except Exception:
            pass
        if not daten.get("vapid_pem"):
            # Beim ersten Mal selbst erzeugen. Der private Schlüssel liegt
            # im privaten Datensatz — dieselbe Vertrauensgrenze wie die
            # Protokolle, und niemand muss ein Geheimnis von Hand setzen.
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import serialization
            k = ec.generate_private_key(ec.SECP256R1())
            daten["vapid_pem"] = k.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()).decode()
            _push_sichern(daten)
        _push_cache["daten"] = daten
    return _push_cache["daten"]


def _push_sichern(daten):
    api.upload_file(path_or_fileobj=json.dumps(daten).encode("utf-8"),
                    path_in_repo=PUSH_DATEI, repo_id=DATEN_REPO,
                    repo_type="dataset", commit_message="Push aktualisiert")
    _push_cache["daten"] = daten


def _push_oeffentlich():
    from cryptography.hazmat.primitives import serialization
    k = serialization.load_pem_private_key(
        _push_zustand()["vapid_pem"].encode(), password=None)
    roh = k.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint)
    import base64
    return base64.urlsafe_b64encode(roh).decode().rstrip("=")


_vapid_pfad = {"wert": None}


def _vapid_datei():
    """Schreibt den privaten Schlüssel einmalig als Datei.

    pywebpush nimmt für `vapid_private_key` einen Dateipfad oder
    base64-kodiertes DER — eine PEM-Zeichenkette weist es mit „Could not
    deserialize key data“ zurück. Der Dateipfad ist die dokumentierte und
    eindeutige Form; die Datei liegt nur im flüchtigen Space-Dateisystem.
    """
    if not _vapid_pfad["wert"] or not os.path.exists(_vapid_pfad["wert"]):
        pfad = os.path.join(tempfile.gettempdir(), "vapid_privat.pem")
        with open(pfad, "w", encoding="utf-8") as fh:
            fh.write(_push_zustand()["vapid_pem"])
        os.chmod(pfad, 0o600)
        _vapid_pfad["wert"] = pfad
    return _vapid_pfad["wert"]


def _dienst(abo):
    """Welcher Zustelldienst — hilft, Apple von anderen zu unterscheiden."""
    try:
        return (abo.get("endpoint", "").split("/")[2]) or "?"
    except Exception:
        return "?"


def _push_bereit():
    """Ist die Versandbibliothek überhaupt da? -> (ja, Grund)

    pywebpush zieht http-ece nach, das eine Erweiterung baut. Scheitert
    das beim Aufbau des Space, wäre Push still tot — der Fehler fiele
    erst auf, wenn eine erwartete Nachricht ausbleibt. Deshalb abfragbar.
    """
    try:
        import pywebpush  # noqa: F401
        return True, "bereit"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _push_senden(titel, text):
    """Verschickt an alle angemeldeten Geräte. -> (zugestellt, Fehler)"""
    try:
        from pywebpush import webpush, WebPushException
    except Exception as e:
        return 0, [f"Versandbibliothek fehlt: {e}"]
    z = _push_zustand()
    uebrig, geaendert, zugestellt, fehler = [], False, 0, []
    for abo in z.get("abos", []):
        try:
            # Zwei Dinge, die beim ersten echten Versand aufgefallen sind:
            # ohne Zeitgrenze kann der Aufruf hängen, bis die HTTP-Anfrage
            # des Browsers aufgibt — dann sieht der Nutzer nur „An error
            # occurred“ und weiß nichts. Und der sub-Anspruch muss eine
            # erreichbare Adresse sein; erfundene .invalid-Domains weisen
            # Zustelldienste zurück.
            webpush(subscription_info=abo,
                    data=json.dumps({"titel": titel, "text": text}),
                    vapid_private_key=_vapid_datei(),
                    vapid_claims={"sub": FRONTEND},
                    timeout=10)
            uebrig.append(abo)
            zugestellt += 1
        except WebPushException as e:
            # 404/410 heißt: Gerät hat abgemeldet. Eintrag darf weg.
            antwort = getattr(e, "response", None)
            code = getattr(antwort, "status_code", None)
            rumpf = ""
            try:
                rumpf = (antwort.text or "")[:120]
            except Exception:
                pass
            if code in (404, 410):
                geaendert = True
            else:
                uebrig.append(abo)
            fehler.append(f"{_dienst(abo)} {code or '?'}: {rumpf or e}"[:200])
        except Exception as e:
            uebrig.append(abo)
            fehler.append(f"{_dienst(abo)} {type(e).__name__}: {e}"[:200])
    if geaendert:
        z["abos"] = uebrig
        try:
            _push_sichern(z)
        except Exception:
            traceback.print_exc()
    return zugestellt, fehler


def push_schluessel_api(key):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    try:
        return {"ok": True, "schluessel": _push_oeffentlich()}
    except Exception as e:
        return {"ok": False, "fehler": str(e)}


def push_stand_api(key):
    """Sagt ohne Netzverkehr, wie es um Push steht.

    Bewusst getrennt vom Versand: hängt der Versand, liefert diese
    Auskunft trotzdem sofort und man sieht, woran es nicht liegt.
    """
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    bereit, grund = _push_bereit()
    try:
        z = _push_zustand()
        return {"ok": True, "bibliothek": grund,
                "geraete": [_dienst(a) for a in z.get("abos", [])],
                "schluessel": bool(z.get("vapid_pem"))}
    except Exception as e:
        return {"ok": False, "bibliothek": grund, "fehler": str(e)}


def push_pruefen_api(key):
    """Schickt eine Probe-Benachrichtigung und sagt, was dabei geschah."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    bereit, grund = _push_bereit()
    try:
        geraete = len(_push_zustand().get("abos", []))
    except Exception as e:
        return {"ok": False, "fehler": f"Push-Zustand nicht lesbar: {e}"}
    if not bereit:
        return {"ok": False, "bibliothek": grund, "geraete": geraete,
                "fehler": "Versandbibliothek fehlt im Space."}
    if not geraete:
        return {"ok": False, "bibliothek": grund, "geraete": 0,
                "fehler": "Kein Gerät angemeldet."}
    try:
        zugestellt, fehler = _push_senden(
            "Probe", "Wenn du das liest, funktionieren Benachrichtigungen.")
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "bibliothek": grund, "geraete": geraete,
                "fehler": f"{type(e).__name__}: {e}"}
    return {"ok": zugestellt > 0, "bibliothek": grund, "geraete": geraete,
            "zugestellt": zugestellt, "meldungen": fehler}


def push_anmelden_api(key, abo):
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    if not isinstance(abo, dict) or not abo.get("endpoint"):
        return {"ok": False, "fehler": "Ungültige Anmeldung."}
    try:
        z = _push_zustand()
        abos = [a for a in z.get("abos", [])
                if a.get("endpoint") != abo["endpoint"]]
        abos.append(abo)
        z["abos"] = abos[-10:]      # ein paar Geräte reichen
        _push_sichern(z)
    except Exception as e:
        return {"ok": False, "fehler": str(e)}
    return {"ok": True, "geraete": len(z["abos"])}


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
        turns, stats, overlaps, abdruecke = _diarize_gpu(s["wav"], num_speakers)
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
             abdruecke=abdruecke, abgleich_chunks=[], phase="haupt",
             sprache=sprache, fa_zeit=0.0, fa_wort=0, fertig=set(),
             fenster=_fenstergrenzen(turns, s["dauer"]))
    s["qualitaet"] = aufnahme_urteil(_pegel_messen(s["wav"], turns) or {})

    # Bekannte Stimmen wiedererkennen: dieselben Personen sitzen Woche für
    # Woche im selben Meeting.
    s["vorschlag"] = stimmen_zuordnen(abdruecke, _stimmen_laden())

    fa_abschnitte = max(1, math.ceil(s["dauer"] / FA_FENSTER))
    asr_abschnitte = max(1, len(s["fenster"]) - 1)
    if s.get("worte"):
        # Erst das Transkript einpassen, danach zum Abgleich selbst
        # erkennen. Zwei unabhängige Erkennungen sind der einzige
        # kostenlose Hinweis darauf, wo der Text unsicher ist.
        abschnitte = fa_abschnitte + asr_abschnitte
    else:
        abschnitte = asr_abschnitte
    return {"ok": True, "sprecher": len(stats), "sprache": sprache,
            "abschnitte": abschnitte, "qualitaet": s["qualitaet"],
            "vorschlag": s["vorschlag"]}


def transkribieren_api(key, sid, index):
    """Schritt 3: nächsten Abschnitt verarbeiten (eigene Anfrage je Abschnitt)."""
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    s = _sitzung(sid)
    if not s or "fenster" not in s:
        return {"ok": False, "fehler": "Sitzung abgelaufen. Bitte erneut starten."}

    if s.get("worte") and s.get("phase") == "haupt":
        try:
            weiter = _align_schritt(s)
        except Exception as e:
            traceback.print_exc()
            return {"ok": False,
                    "fehler": f"Transkript einpassen fehlgeschlagen: "
                              f"{type(e).__name__}: {e}"}
        if not weiter:
            # Eingepasst. Jetzt dieselbe Aufnahme selbst erkennen, damit
            # sich die beiden Ergebnisse gegenseitig prüfen können.
            s["phase"] = "abgleich"
            s["fertig"] = set()
            weiter = True
        return {"ok": True, "weiter": weiter, "schritt": "einpassen",
                "fortschritt": round(min(1.0, s["fa_zeit"] / max(s["dauer"], 1)), 2)}

    i = int(index)
    if not 0 <= i < len(s["fenster"]) - 1:
        return {"ok": False, "fehler": "Ungültiger Abschnitt."}
    if i in s.setdefault("fertig", set()):
        # Bereits verarbeitet — Wiederholung nach Abbruch darf nichts doppeln
        return {"ok": True, "index": i, "abschnitte": len(s["fenster"]) - 1,
                "weiter": i + 1 < len(s["fenster"]) - 1}
    ziel_liste = ("abgleich_chunks" if s.get("phase") == "abgleich"
                  else "chunks")
    try:
        s[ziel_liste] += _transkribiere_gpu(s["wav"], s["fenster"][i],
                                            s["fenster"][i + 1],
                                            s.get("sprache"),
                                            s.get("kontext", ""))
    except Exception as e:
        traceback.print_exc()
        return {"ok": False,
                "fehler": f"Transkription fehlgeschlagen: {type(e).__name__}: {e}"}
    s["fertig"].add(i)
    return {"ok": True, "index": i, "abschnitte": len(s["fenster"]) - 1,
            "weiter": i + 1 < len(s["fenster"]) - 1}


def _daten_sichern(s):
    if "daten" not in s:
        # Zwei unabhängige Erkennungen: wo sie sich widersprechen, ist der
        # Text unsicher. Der Wortlaut der Vorlage bleibt stehen — markiert
        # wird nur, was bestritten ist.
        strittig_idx = ()
        if s.get("abgleich_chunks"):
            haupt = [c["text"] for c in sorted(s["chunks"],
                                               key=lambda c: c["start"])]
            zweit = [c["text"] for c in sorted(s["abgleich_chunks"],
                                               key=lambda c: c["start"])]
            s["strittig"] = abgleich(haupt, zweit)
            strittig_idx = {x["index"] for x in s["strittig"]}
        s["chunks"] = _konfidenz_markieren(
            sorted(s["chunks"], key=lambda c: c["start"]), strittig_idx)
        daten = _zusammenfuegen(s["turns"], s["stats"], s["overlaps"],
                                s["chunks"], s["dauer"], s.get("sprache"))
        # Erst nach der Zuordnung korrigieren: die Zeitstempel stammen aus
        # dem Audio und dürfen von einer Textänderung nicht berührt werden.
        # Ein mitgeliefertes Transkript bleibt unangetastet — es ist die
        # Vorlage, nicht die Vermutung.
        if not s.get("worte"):
            g = _glossar_laden()
            ziele = _korrekturziele(g, s.get("eigener", ""))
            daten["segmente"], daten["korrekturen"] = _korrigieren(
                daten["segmente"], ziele, g.get("festen"))
        else:
            daten["korrekturen"] = []
        daten["qualitaet"] = s.get("qualitaet")
        daten["strittig"] = s.get("strittig", [])
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
            "sprecher_bekannt": glossar["sprecher"],
            # Damit die App die Stimmen beim Speichern fortschreiben kann
            # und benannte Sprecher gleich vorausgefüllt sind.
            "abdruecke": s.get("abdruecke") or {},
            "vorschlag": s.get("vorschlag") or {},
            "qualitaet": s.get("qualitaet")}


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
        # Und die Stimme dazu: beim nächsten Mal erkennt die App die
        # Person wieder, statt sie erneut nummerieren zu lassen.
        abdruecke = (analyse.get("abdruecke") or {})
        if abdruecke:
            try:
                _stimmen_sichern(stimmen_auffrischen(
                    _stimmen_laden(frisch=True), abdruecke, eigene))
            except Exception:
                traceback.print_exc()
    return {"ok": True, "pfad": pfad, "markdown": md_text,
            "neue_begriffe": neue}


def lernen_api(key, falsch, richtig):
    """Nimmt eine vom Nutzer bestätigte Korrektur auf.

    Bis hierher lernte das System ausschließlich aus der eigenen Ausgabe —
    ein Kreis ohne Wahrheit von außen, der nur die eigenen Fehler
    verstärken kann. Dies ist der einzige Eingang für echtes Wissen: was
    der Mensch beim Lesen als falsch erkannt und richtiggestellt hat.
    """
    if not _pruefe(key):
        return {"ok": False, "fehler": "Ungültiger Zugangsschlüssel."}
    falsch = (falsch or "").strip()
    richtig = (richtig or "").strip()
    if not falsch or not richtig or falsch.lower() == richtig.lower():
        return {"ok": False, "fehler": "Nichts zu lernen."}
    if len(richtig) > 80 or " " in richtig:
        return {"ok": False, "fehler": "Bitte ein einzelnes Wort."}
    try:
        g = _glossar_laden(frisch=True)
        festen = dict(g.get("festen") or {})
        festen[falsch.lower()] = richtig
        zaehler = dict(g["zaehler"])
        # Das richtige Wort zählt ab jetzt als bestätigter Begriff, damit
        # auch ähnliche Verhörer darauf gezogen werden.
        zaehler[richtig] = max(zaehler.get(richtig, 0), KORREKTUR_MIN_ZAEHLER)
        _glossar_speichern(zaehler, g["sprecher"], festen)
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "fehler": f"Speichern fehlgeschlagen: {e}"}
    return {"ok": True, "falsch": falsch, "richtig": richtig}


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
        b_starten = gr.Button("Im Hintergrund verarbeiten")
        b_starten.click(starten_api,
                        [key, sid, gr.Number(value=0), gr.Textbox(value="german")],
                        gr.JSON(), api_name="starten")
        gr.Button("Auftragsstand").click(
            auftrag_api, [key, gr.Textbox(label="Auftrag")], gr.JSON(),
            api_name="auftrag")
        gr.Button("Push-Schlüssel").click(
            push_schluessel_api, [key], gr.JSON(), api_name="push_schluessel")
        gr.Button("Push-Zustand").click(
            push_stand_api, [key], gr.JSON(), api_name="push_stand")
        gr.Button("Push prüfen").click(
            push_pruefen_api, [key], gr.JSON(), api_name="push_pruefen")
        gr.Button("Push anmelden").click(
            push_anmelden_api, [key, gr.JSON(label="Abo")], gr.JSON(),
            api_name="push_anmelden")
        falsch_feld = gr.Textbox(label="Falsch erkannt")
        richtig_feld = gr.Textbox(label="Richtig")
        gr.Button("Korrektur merken").click(
            lernen_api, [key, falsch_feld, richtig_feld], gr.JSON(),
            api_name="lernen")
    with gr.Tab("Verlauf"):
        b_verlauf = gr.Button("Verlauf laden")
        out_verlauf = gr.JSON()
        b_verlauf.click(verlauf_api, [key], out_verlauf, api_name="verlauf")
        pfad = gr.Textbox(label="Pfad (aufnahmen/....md)")
        b_eintrag = gr.Button("Eintrag laden")
        out_eintrag = gr.JSON()
        b_eintrag.click(eintrag_api, [key, pfad], out_eintrag, api_name="eintrag")

demo.launch(ssr_mode=False)
