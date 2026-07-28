"""Tests der Fachlogik — ohne Modelle, GPU oder Netz.

Deckt die Stellen ab, an denen in der Entwicklung tatsächlich Fehler
aufgetreten sind: Fensterung langer Aufnahmen, Sprecherzuordnung,
Markdown-Erzeugung, Glossar-Rangfolge und die Pfadprüfung.
"""
import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf-space"))

import kern  # noqa: E402


# ---------- Fensterung ----------

def test_kurzes_audio_ergibt_ein_fenster():
    grenzen = kern._fenstergrenzen([{"start": 0.0, "ende": 20.0}], 23.4)
    assert grenzen == [0.0, 23.4]


def test_lange_aufnahme_wird_lueckenlos_zerlegt():
    turns = [{"start": i * 20.0, "ende": i * 20.0 + 18} for i in range(180)]
    gesamt = 3600.0
    grenzen = kern._fenstergrenzen(turns, gesamt)
    assert grenzen[0] == 0.0 and grenzen[-1] == gesamt
    assert all(b > a for a, b in zip(grenzen, grenzen[1:])), "nicht monoton"
    assert max(b - a for a, b in zip(grenzen, grenzen[1:])) <= kern.FENSTER + 120


def test_schnittpunkte_liegen_auf_sprecherwechseln():
    turns = [{"start": i * 20.0, "ende": i * 20.0 + 18} for i in range(120)]
    grenzen = kern._fenstergrenzen(turns, 2400.0)
    starts = {t["start"] for t in turns}
    assert all(g in starts for g in grenzen[1:-1])


# ---------- Sprecherzuordnung ----------

TURNS = [{"start": 0.0, "ende": 5.0, "label": "A"},
         {"start": 5.5, "ende": 9.0, "label": "B"}]


@pytest.mark.parametrize("zeitpunkt,erwartet", [
    (1.0, "A"), (4.9, "A"), (6.0, "B"), (8.9, "B"),
    (5.2, "A"),      # in der Lücke: nächstgelegener Sprecher
    (99.0, "B"),     # weit dahinter: letzter Sprecher
])
def test_sprecher_zum_zeitpunkt(zeitpunkt, erwartet):
    assert kern._sprecher_bei(TURNS, zeitpunkt) == erwartet


def test_ohne_sprecher_kein_treffer():
    assert kern._sprecher_bei([], 1.0) is None


def test_worte_werden_zu_sprecherabschnitten_gruppiert():
    chunks = [{"start": 0.5, "ende": 1.0, "text": "Guten"},
              {"start": 1.1, "ende": 1.6, "text": "Tag"},
              {"start": 6.0, "ende": 6.4, "text": "Hallo"}]
    daten = kern._zusammenfuegen(TURNS, {"A": {"dauer": 5.0, "turns": 1},
                                         "B": {"dauer": 3.5, "turns": 1}},
                                 [], chunks, 9.0)
    assert [s["label"] for s in daten["segmente"]] == ["A", "B"]
    assert daten["segmente"][0]["text"] == "Guten Tag"
    assert daten["segmente"][1]["text"] == "Hallo"


def test_ohne_erkannten_text_bleibt_die_sprecherstruktur():
    daten = kern._zusammenfuegen(TURNS, {"A": {"dauer": 5.0, "turns": 1}},
                                 [], [], 9.0)
    assert len(daten["segmente"]) == 2
    assert all(s["text"] is None for s in daten["segmente"])


# ---------- Markdown ----------

def _beispiel_daten():
    return {"dauer": 9.0, "sprache": "german",
            "stats": {"A": {"dauer": 5.0, "turns": 1},
                      "B": {"dauer": 3.5, "turns": 1}},
            "overlaps": [{"start": 4.9, "ende": 5.1, "wer": ["A", "B"]}],
            "segmente": [
                {"start": 0.5, "ende": 1.6, "label": "A", "text": "Guten Tag"},
                {"start": 6.0, "ende": 6.4, "label": "B", "text": "Hallo"}]}


def test_markdown_enthaelt_frontmatter_und_abschnitte():
    md = kern._markdown(_beispiel_daten(), "aufnahme.m4a",
                        datetime.datetime(2026, 7, 28, 9, 5))
    for erwartet in ["---", "titel: Aufnahme 2026-07-28 09:05", "sprache: german",
                     "## Redeanteile", "## Protokoll", "## Segmente",
                     "## Gleichzeitiges Sprechen"]:
        assert erwartet in md, f"fehlt: {erwartet}"
    fm, rumpf = kern._frontmatter(md)
    assert fm["sprecher"] == 2 and fm["dauer_s"] == 9.0
    assert rumpf.lstrip().startswith("# Aufnahme")


def test_eigene_sprechernamen_ersetzen_die_standardnamen():
    md = kern._markdown(_beispiel_daten(), "a.m4a",
                        datetime.datetime(2026, 7, 28, 9, 5), {"A": "Nils"})
    assert "Nils" in md and "Sprecher 1" not in md
    assert "Sprecher 2" in md, "unbenannte Sprecher behalten den Standardnamen"


def test_frontmatter_vertraegt_text_ohne_kopf():
    fm, rumpf = kern._frontmatter("# Nur Text")
    assert fm == {} and rumpf == "# Nur Text"


# ---------- Glossar ----------

def test_glossar_faellt_haeufigstes_zuerst():
    zaehler = {"Kita": 3, "Ausflug": 9, "Gruppe": 3}
    assert kern._rangfolge(zaehler)[0] == "Ausflug"
    assert kern._rangfolge(zaehler)[1:] == ["Gruppe", "Kita"]  # alphabetisch


def test_glossar_schreiben_und_lesen_ist_verlustfrei():
    zaehler = {"Entwicklungsgespräch": 12, "Kita": 4}
    text, geordnet = kern.glossar_schreiben(zaehler, ["Nils", "Anna"])
    assert geordnet == ["Entwicklungsgespräch", "Kita"]
    gelesen, sprecher = kern.glossar_lesen(text)
    assert gelesen == zaehler and sprecher == ["Nils", "Anna"]


def test_glossar_wird_gedeckelt():
    zaehler = {f"Wort{i}": i for i in range(1, kern.GLOSSAR_MAX + 50)}
    _, geordnet = kern.glossar_schreiben(zaehler, [])
    assert len(geordnet) == kern.GLOSSAR_MAX


def test_nur_wiederkehrende_begriffe_werden_gezaehlt():
    segmente = [{"text": "Das Entwicklungsgespräch mit Hedi war gut."},
                {"text": "Beim Entwicklungsgespräch hat Hedi viel erzählt."},
                {"text": "Genau, das war schon so."}]
    zaehler = kern._begriffe_zaehlen(segmente)
    assert zaehler == {"Entwicklungsgespräch": 2, "Hedi": 2}, zaehler


def test_kontext_bleibt_kurz_und_priorisiert():
    glossar = {"sprecher": ["Nils"],
               "begriffe": [f"Begriff{i}" for i in range(200)]}
    kontext = kern._kontext_bauen("Kita", glossar)
    assert kontext.startswith("Kita, Nils"), kontext
    assert len(kontext) <= 380


# ---------- Umschrift fürs Alignment ----------

@pytest.mark.parametrize("wort,erwartet", [
    ("Müller", "muller"), ("Straße", "strasse"), ("Café", "cafe"),
    ("Guten!", "guten"), ("z.B.", "zb"), ("123", ""),
])
def test_umschrift(wort, erwartet):
    assert kern._romanisieren(wort) == erwartet


# ---------- Pfadprüfung ----------

@pytest.mark.parametrize("pfad", [
    "aufnahmen/2026-07-28_090500.md",
])
def test_erlaubte_pfade(pfad):
    assert kern.pfad_erlaubt(pfad)


@pytest.mark.parametrize("pfad", [
    "glossar.md", "aufnahmen/../glossar.md", "/etc/passwd",
    "aufnahmen/x.txt", "aufnahmen/sub/dir.md", "", None, 42,
])
def test_abgelehnte_pfade(pfad):
    assert not kern.pfad_erlaubt(pfad)
