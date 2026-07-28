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


def test_schnittpunkte_liegen_in_stille():
    """Schärfer als die frühere Zusage.

    Vorher genügte ein Sprecherwechsel. Der ist aber keine Pause, wenn der
    nächste einsetzt, während der vorige noch spricht — dann fällt der
    Schnitt mitten ins Wort. Verlangt wird jetzt, dass zum Schnittzeitpunkt
    wirklich niemand redet.
    """
    turns = [{"start": i * 20.0, "ende": i * 20.0 + 18} for i in range(120)]
    grenzen = kern._fenstergrenzen(turns, 2400.0)
    for g in grenzen[1:-1]:
        spricht = [t for t in turns if t["start"] < g < t["ende"]]
        assert not spricht, f"Schnitt bei {g} fällt mitten in einen Beitrag"


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


# ---------- Abgleich zweier Erkennungen ----------

def test_einigkeit_erzeugt_keine_markierung():
    assert kern.abgleich(["guten", "tag"], ["Guten", "Tag!"]) == []


def test_widerspruch_wird_benannt():
    s = kern.abgleich(["die", "martha", "kam"], ["die", "marta", "kam"])
    assert len(s) == 1
    assert s[0]["index"] == 1 and s[0]["wort"] == "martha"
    assert s[0]["gehoert"] == "marta"


def test_fehlendes_wort_gilt_als_strittig():
    s = kern.abgleich(["a", "bcd", "e"], ["a", "e"])
    assert [x["wort"] for x in s] == ["bcd"] and s[0]["gehoert"] is None


def test_der_wortlaut_wird_nie_veraendert():
    # Der Abgleich markiert, er ersetzt nicht — keine der beiden Quellen
    # ist die Wahrheit.
    vorlage = ["die", "martha", "kam"]
    kern.abgleich(vorlage, ["die", "marta", "kam"])
    assert vorlage == ["die", "martha", "kam"]


def test_ohne_zweite_erkennung_keine_behauptung():
    assert kern.abgleich(["a", "b"], []) == []


# ---------- Konfidenz ----------

def test_schwacher_alignmentwert_gilt_als_unsicher():
    c = kern._konfidenz_markieren([{"text": "x", "wert": 0.2},
                                   {"text": "y", "wert": 0.95}])
    assert c[0]["unsicher"] and not c[1]["unsicher"]


def test_widerspruch_macht_unsicher_trotz_gutem_wert():
    c = kern._konfidenz_markieren([{"text": "x", "wert": 0.99}], {0})
    assert c[0]["unsicher"], "der Widerspruch der zweiten Erkennung zählt auch"


def test_fehlender_wert_macht_nicht_unsicher():
    # Keine Angabe ist kein Verdacht; sonst wäre alles grau.
    assert not kern._konfidenz_markieren([{"text": "x"}])[0]["unsicher"]


def test_unsicherheit_landet_als_position_im_segment():
    chunks = [{"start": 0.0, "ende": 1.0, "text": "Guten", "unsicher": False},
              {"start": 1.0, "ende": 2.0, "text": "Tag", "unsicher": True}]
    daten = kern._zusammenfuegen([_turn(0, 5)], {"A": {"dauer": 5, "turns": 1}},
                                 [], chunks, 5.0)
    seg = daten["segmente"][0]
    assert seg["text"] == "Guten Tag", "der Wortlaut bleibt unversehrt"
    assert seg["unsicher"] == [1], "markiert wird über die Position"


# ---------- Stimmabdrücke ----------

def test_gleiche_stimme_wird_wiedererkannt():
    bekannt = {"Martha": [1.0, 0.0, 0.0], "Nils": [0.0, 1.0, 0.0]}
    zu = kern.stimmen_zuordnen({"A": [0.98, 0.02, 0.0]}, bekannt)
    assert zu == {"A": "Martha"}


def test_fremde_stimme_bleibt_unbenannt():
    bekannt = {"Martha": [1.0, 0.0, 0.0]}
    assert kern.stimmen_zuordnen({"A": [0.0, 0.0, 1.0]}, bekannt) == {}


def test_zwei_aehnliche_stimmen_werden_nicht_geraten():
    # Ohne deutlichen Vorsprung lieber gar kein Name: ein falscher Name
    # im Protokoll fällt beim Lesen kaum auf.
    bekannt = {"Anna": [1.0, 0.0], "Anne": [0.999, 0.045]}
    assert kern.stimmen_zuordnen({"A": [1.0, 0.02]}, bekannt) == {}


def test_ein_name_wird_nicht_zweimal_vergeben():
    bekannt = {"Martha": [1.0, 0.0, 0.0]}
    zu = kern.stimmen_zuordnen({"A": [1.0, 0.0, 0.0], "B": [0.99, 0.01, 0.0]},
                               bekannt)
    assert list(zu.values()) == ["Martha"]


def test_abdruck_wird_gemittelt_nicht_ueberschrieben():
    # Eine Aufnahme mit Schnupfen darf die gespeicherte Stimme nicht
    # umwerfen.
    neu = kern.stimmen_auffrischen({"Martha": [1.0, 0.0]},
                                   {"A": [0.0, 1.0]}, {"A": "Martha"})
    assert neu["Martha"][0] > 0.5, f"zu stark verschoben: {neu['Martha']}"


def test_unbekannter_name_wird_neu_angelegt():
    neu = kern.stimmen_auffrischen({}, {"A": [1.0, 0.0]}, {"A": "Hedi"})
    assert neu["Hedi"] == [1.0, 0.0]


# ---------- Aufnahmequalität ----------

def test_sauberes_signal_gilt_als_gut():
    u = kern.aufnahme_urteil({"sprache_db": -20.0, "rausch_db": -55.0})
    assert u["stufe"] == "gut" and u["rat"] == []


def test_zu_grosser_abstand_wird_erkannt_und_erklaert():
    u = kern.aufnahme_urteil({"sprache_db": -30.0, "rausch_db": -38.0})
    assert u["stufe"] == "schlecht"
    assert any("näher" in r for r in u["rat"]), u["rat"]


def test_uebersteuerung_wird_gemeldet():
    u = kern.aufnahme_urteil({"sprache_db": -12.0, "rausch_db": -50.0,
                              "uebersteuert": 0.03})
    assert any("übersteuert" in r for r in u["rat"]), u["rat"]


def test_urteil_ohne_messwerte_stuerzt_nicht_ab():
    assert kern.aufnahme_urteil({})["stufe"] in ("gut", "grenzwertig",
                                                 "schlecht")


# ---------- Bestätigte Korrekturen ----------

def test_bestaetigte_korrektur_gilt_ohne_aehnlichkeitspruefung():
    # Vom Nutzer bestätigt heißt: unbedingt. Auch wenn die Wörter weit
    # auseinanderliegen und der Anfangsbuchstabe nicht passt.
    neu, prot = kern._korrigieren(_seg("Wir sprachen mit Kolleschin"), [],
                                  {"kolleschin": "Kollegin"})
    assert "Kollegin" in neu[0]["text"]
    assert prot[0]["nachher"] == "Kollegin"


def test_bestaetigte_korrektur_schlaegt_die_aehnlichkeit():
    neu, _ = kern._korrigieren(_seg("Marta kam"), ["Martina"],
                               {"marta": "Martha"})
    assert "Martha" in neu[0]["text"]


# ---------- Messung ----------

def test_gleicher_text_hat_keine_fehler():
    e = kern.wortfehlerrate("Guten Tag, wie geht es Ihnen?",
                            "guten tag wie geht es ihnen")
    assert e["rate"] == 0.0, "Satzzeichen und Groß/Klein dürfen nicht zählen"


def test_umlaute_werden_vereinheitlicht():
    assert kern.wortfehlerrate("Gespräch über Größe",
                               "Gespraech ueber Groesse")["rate"] == 0.0


def test_die_drei_fehlerarten_werden_getrennt():
    e = kern.wortfehlerrate("a b c d", "a x c d e")
    assert e["ersetzungen"] == 1 and e["einfuegungen"] == 1
    assert e["loeschungen"] == 0
    assert e["rate"] == 0.5, "zwei Fehler auf vier Wörter"


def test_fehlende_woerter_zaehlen():
    e = kern.wortfehlerrate("a b c d", "a d")
    assert e["loeschungen"] == 2 and e["rate"] == 0.5


def test_verwechslungen_werden_benannt():
    e = kern.wortfehlerrate("die martha kam", "die marta kam")
    assert ("martha", "marta") in dict(e["haeufigste"])


def test_leere_vorlage_bricht_nicht():
    assert kern.wortfehlerrate("", "irgendwas")["rate"] == 0.0


def test_protokolltext_loest_das_markdown_auf():
    md = ("---\ntitel: x\n---\n\n# Aufnahme\n\n## Redeanteile\n\n"
          "| Sprecher | Anteil |\n|---|---|\n| Sprecher 1 | 58 % |\n\n"
          "## Protokoll\n\n"
          "**Sprecher 1** (0:00–0:02): Guten Tag\n\n"
          "**Sprecher 2** (0:06–0:07): Hallo zusammen\n\n"
          "## Segmente\n\n| Start | Ende |\n|---|---|\n| 0:00 | 0:02 |\n")
    text = kern.protokoll_text(md)
    assert text == "Guten Tag Hallo zusammen", text
    # Entscheidend: nichts aus Kopfdaten oder Tabellen darf mitkommen,
    # sonst misst man die Vorlage gegen Tabellenzeilen.
    for fremd in ("Sprecher 1", "58", "titel", "0:00"):
        assert fremd not in text, f"{fremd!r} gehört nicht in den Wortlaut"


# ---------- Fensterschnitte ----------

def _turn(a, b, label="A"):
    return {"start": a, "ende": b, "label": label}


def test_pausen_beruecksichtigen_ueberlappung():
    # Ein Sprecherwechsel ist keine Pause, wenn der nächste einsetzt,
    # während der vorige noch redet.
    turns = [_turn(0, 10, "A"), _turn(8, 20, "B")]
    assert kern._sprechpausen(turns) == []


def test_echte_luecke_wird_gefunden():
    turns = [_turn(0, 10, "A"), _turn(15, 20, "B")]
    assert kern._sprechpausen(turns) == [(10, 15)]


def test_schnitt_landet_in_der_laengsten_stille():
    gesamt = kern.FENSTER * 2
    ziel = kern.FENSTER
    turns = [_turn(0, ziel - 40),
             _turn(ziel - 39, ziel - 20),      # 1 s Lücke davor
             _turn(ziel + 10, gesamt)]         # 30 s Lücke davor
    grenzen = kern._fenstergrenzen(turns, gesamt)
    mitte = (ziel - 20 + ziel + 10) / 2
    assert abs(grenzen[1] - mitte) < 0.01, \
        f"Schnitt nicht in der grossen Stille: {grenzen[1]}"


def test_ohne_pause_wird_am_sprecherwechsel_getrennt():
    gesamt = kern.FENSTER * 2
    # Durchgehend belegt, aber mit einem Wechsel nahe der Zielmarke.
    turns = [_turn(0, kern.FENSTER + 5, "A"),
             _turn(kern.FENSTER + 3, gesamt, "B")]
    grenzen = kern._fenstergrenzen(turns, gesamt)
    assert grenzen[1] == kern.FENSTER + 3


def test_kurze_aufnahme_bleibt_ein_stueck():
    assert kern._fenstergrenzen([], 60.0) == [0.0, 60.0]


def test_grenzen_sind_aufsteigend_und_vollstaendig():
    gesamt = kern.FENSTER * 3.5
    turns = [_turn(i * 20.0, i * 20.0 + 15.0) for i in range(int(gesamt / 20))]
    g = kern._fenstergrenzen(turns, gesamt)
    assert g[0] == 0.0 and g[-1] == gesamt
    assert all(b > a for a, b in zip(g, g[1:])), f"nicht aufsteigend: {g}"


# ---------- Halluzinationen ----------

def test_untertitel_abspann_wird_verworfen():
    for erfunden in ("Untertitel der Amara.org-Community",
                     "Untertitelung im Auftrag des ZDF für funk, 2017",
                     "Vielen Dank.",
                     "Danke fürs Zuschauen!",
                     "Abonniert den Kanal für mehr Videos"):
        assert kern._ist_halluzination(erfunden), erfunden


def test_festgefahrene_wiederholung_wird_verworfen():
    assert kern._ist_halluzination("ja ja ja ja ja ja ja ja")


def test_echter_satz_bleibt():
    for echt in ("Vielen Dank für das Gespräch, das war sehr aufschlussreich.",
                 "Ja, genau, das sehe ich auch so.",
                 "Hedi sucht momentan stärker den Kontakt zu den Erwachsenen."):
        assert not kern._ist_halluzination(echt), echt


def test_wiederholung_in_normaler_rede_bleibt():
    # Doppelungen sind in gesprochener Sprache normal und dürfen nicht
    # als Halluzination durchgehen.
    assert not kern._ist_halluzination("Das ist ist wirklich sehr sehr gut")


# ---------- Glossar-Korrektur ----------

def _seg(text):
    return [{"start": 0.0, "ende": 1.0, "label": "A", "text": text}]


def test_knapp_daneben_wird_zurechtgerueckt():
    neu, prot = kern._korrigieren(_seg("Also Marta hat gesagt"), ["Martha"])
    assert neu[0]["text"] == "Also Martha hat gesagt"
    assert prot == [{"vorher": "Marta", "nachher": "Martha", "anzahl": 1}]


def test_zeitstempel_bleiben_unberuehrt():
    # Die Zeiten stammen aus dem Audio; eine Textkorrektur darf sie nicht
    # anfassen, sonst wandert der Text zum falschen Sprecher.
    vorher = _seg("Marta")
    neu, _ = kern._korrigieren(vorher, ["Martha"])
    assert neu[0]["start"] == 0.0 and neu[0]["ende"] == 1.0
    assert neu[0]["label"] == "A"


def test_gelaeufiges_wort_bleibt_stehen():
    # „Augen“ darf nicht zu „Auge“ werden, nur weil jemand das ins
    # Glossar geschrieben hat.
    neu, prot = kern._korrigieren(_seg("in ihren Augen"), ["Auge"])
    assert neu[0]["text"] == "in ihren Augen" and prot == []


def test_mehrdeutiges_bleibt_unangetastet():
    # Gleich nah an zwei Zielen heißt: die Korrektur wäre geraten.
    neu, prot = kern._korrigieren(_seg("Herr Mann"), ["Mann", "Wann"])
    assert "Mann" in neu[0]["text"] and prot == []


def test_zu_weit_entfernt_wird_nicht_angefasst():
    neu, prot = kern._korrigieren(_seg("Der Wagen"), ["Hagen"])
    assert prot == [], "zwei Ersetzungen bei vier Zeichen sind kein Treffer"


def test_kurze_ziele_werden_ignoriert():
    # Bei drei Zeichen liegt zu viel im Umkreis von eins.
    assert kern._korrekturziele({"sprecher": ["Ali"]}) == []


def test_korrektur_ohne_ziele_aendert_nichts():
    vorher = _seg("Beliebiger Text")
    neu, prot = kern._korrigieren(vorher, [])
    assert neu == vorher and prot == []


def test_ziele_bevorzugen_was_der_nutzer_selbst_eintrug():
    ziele = kern._korrekturziele(
        {"sprecher": ["Hedi"], "zaehler": {"Spielverhalten": 9, "Einmalig": 2}},
        eigener="Entwicklungsgespräch, Kita")
    assert "Entwicklungsgespräch" in ziele, "eigene Eingabe muss zählen"
    assert "Hedi" in ziele, "benannte Sprecher müssen zählen"
    assert "Spielverhalten" in ziele, "bestätigte Begriffe zählen"
    assert "Einmalig" not in ziele, \
        "kaum belegte Begriffe duerfen nicht zum Ziel werden"


def test_sperrliste_haelt_allerweltswoerter_aus_dem_glossar():
    # Gemessener Ausgangspunkt: „Ihren“ und „Augen“ führten die Rangfolge
    # an, vor den tatsächlichen Namen.
    text = ("Ich sehe das in Ihren Augen. Die Gruppe war gut. "
            "Und Ihren Augen sieht man die Gruppe an.")
    zaehler = kern._begriffe_zaehlen(_seg(text))
    for junk in ("Ihren", "Augen", "Gruppe"):
        assert junk not in zaehler, f"{junk} gehoert nicht ins Glossar"


def test_nur_am_satzanfang_zaehlt_nicht_als_begriff():
    # Am Satzanfang wird jedes Wort großgeschrieben — das weist es nicht
    # als Substantiv aus.
    zaehler = kern._begriffe_zaehlen(
        _seg("Draußen ist es kalt. Draußen bleibt es kalt."))
    assert "Draußen" not in zaehler


def test_echter_fachbegriff_ueberlebt_beide_siebe():
    zaehler = kern._begriffe_zaehlen(
        _seg("Beim Entwicklungsgespräch ging es ums Spielverhalten. "
             "Das Spielverhalten war Thema im Entwicklungsgespräch."))
    assert zaehler.get("Spielverhalten", 0) >= 2
    assert zaehler.get("Entwicklungsgespräch", 0) >= 2


# ---------- Dekodierung ----------

def test_vorlaufkontext_kommt_nie_ohne_temperaturleiter():
    """Die Kopplung, an der die Qualität hing.

    Die Bibliothek löst den Vorlauf-Kontext erst oberhalb von Temperatur
    0.5. Ohne Rückfall bleibt sie bei None, der Kontext also dauerhaft an —
    und eine begonnene Wiederholung schreibt sich fort. Wer das eine setzt,
    muss das andere mitsetzen.
    """
    gk = kern._decode_optionen("german")
    if gk.get("condition_on_prev_tokens"):
        temps = gk.get("temperature")
        assert isinstance(temps, (list, tuple)) and len(temps) > 1, \
            "Vorlauf-Kontext ohne Temperaturleiter ist die gefährliche Variante"
        assert max(temps) > 0.5, \
            "die Leiter muss über 0.5 reichen, sonst greift die Notbremse nie"


def test_rueckfall_schwellen_sind_gesetzt():
    gk = kern._decode_optionen("german")
    # Ohne diese beiden merkt die Bibliothek gar nicht, dass sie
    # zurückfallen müsste — die Temperaturliste allein tut nichts.
    assert gk["compression_ratio_threshold"] == kern.KOMPRESSION_MAX
    assert gk["logprob_threshold"] == kern.LOGPROB_MIN


def test_temperaturleiter_beginnt_bei_null():
    # Der erste Versuch muss der beste sein: Temperatur 0 mit Strahlsuche.
    # Steigt sie sofort ein, wird auch sauberes Audio gewürfelt.
    assert kern._decode_optionen()["temperature"][0] == 0.0
    assert kern._decode_optionen()["num_beams"] > 1


def test_sprache_wird_nur_gesetzt_wenn_bekannt():
    # Eine leere Sprachangabe darf nicht als Sprache durchgehen, sonst
    # rät Whisper nicht mehr, sondern bekommt Unsinn vorgesetzt.
    assert "language" not in kern._decode_optionen()
    assert "language" not in kern._decode_optionen("")
    assert kern._decode_optionen("german")["language"] == "german"


def test_uebersetzung_bleibt_ausgeschaltet():
    # Das war ein echter Fehler: Whisper kippte mitten im Gespräch
    # ins Englische.
    assert kern._decode_optionen("german")["task"] == "transcribe"


# ---------- Gleichzeitiges Sprechen ----------

def _ue(start, ende, wer=("A", "B")):
    return {"start": start, "ende": ende, "wer": list(wer)}


def test_kurzes_dazwischenreden_wird_verworfen():
    # Ein „mhm“ ist kein gleichzeitiges Sprechen, das jemanden interessiert.
    kurz = [_ue(10.0, 10.3), _ue(20.0, 20.9), _ue(30.0, 30.99)]
    assert kern._dichte_stellen(kurz) == []


def test_eine_lange_stelle_reicht_allein_aus():
    assert len(kern._dichte_stellen([_ue(10.0, 13.0)])) == 1


def test_einzelner_sekundenblitzer_ist_keine_meldung_wert():
    # Über der Mindestdauer, aber unter der Passagen-Schwelle: kostet ein
    # paar Wörter, taugt aber nicht als Warnung.
    assert kern._dichte_stellen([_ue(10.0, 11.2)]) == []


def test_grenzwerte_der_beiden_schwellen():
    # Grenzwerte sind die Stelle, an der solche Filter üblicherweise
    # danebengreifen — beide hier festgenagelt.
    genau = kern._dichte_stellen([_ue(0.0, kern.UEBERLAPP_PASSAGE)])
    assert len(genau) == 1, "genau auf der Schwelle muss zählen"
    knapp = kern.UEBERLAPP_PASSAGE - 0.01
    assert kern._dichte_stellen([_ue(0.0, knapp)]) == []
    # Unterhalb der Mindestdauer summiert sich auch Masse nicht auf:
    # zwanzig „mhm“ ergeben keine unzuverlässige Passage.
    viele_kurze = [_ue(i * 2.0, i * 2.0 + 0.5) for i in range(20)]
    assert kern._dichte_stellen(viele_kurze) == []


def test_nahe_stellen_werden_zu_einer_passage():
    dicht = [_ue(60.0, 62.0), _ue(64.0, 66.0), _ue(69.0, 71.0)]
    passagen = kern._dichte_stellen(dicht)
    assert len(passagen) == 1, "dichte Folge muss eine Passage ergeben"
    assert passagen[0]["start"] == 60.0 and passagen[0]["ende"] == 71.0
    assert passagen[0]["anzahl"] == 3


def test_weit_auseinander_bleibt_getrennt():
    weit = [_ue(60.0, 62.0), _ue(200.0, 202.0)]
    assert len(kern._dichte_stellen(weit)) == 2


def test_beteiligte_einer_passage_werden_vereinigt():
    p = kern._dichte_stellen([_ue(10.0, 12.0, ("A", "B")),
                              _ue(13.0, 15.0, ("B", "C"))])
    assert p[0]["wer"] == ["A", "B", "C"]


def test_unsortierte_eingabe_wird_richtig_gebuendelt():
    p = kern._dichte_stellen([_ue(69.0, 71.0), _ue(60.0, 62.0),
                              _ue(64.0, 66.0)])
    assert len(p) == 1 and p[0]["start"] == 60.0


def test_verschachtelte_stellen_ueberdehnen_die_passage_nicht():
    # Eine kurze Stelle innerhalb einer langen darf das Ende nicht
    # zurückziehen.
    p = kern._dichte_stellen([_ue(10.0, 30.0), _ue(12.0, 14.0)])
    assert len(p) == 1 and p[0]["ende"] == 30.0


def test_zusammenfuegen_filtert_die_rohmessung():
    roh = [_ue(1.0, 1.2), _ue(2.0, 2.1), _ue(4.0, 6.0)]
    daten = kern._zusammenfuegen(TURNS, {"A": {"dauer": 5.0, "turns": 1}},
                                 roh, [], 9.0)
    assert len(daten["overlaps"]) == 1, \
        "Kurzstellen dürfen nicht bis in die Ausgabe durchschlagen"


# ---------- Markdown ----------

def _beispiel_daten():
    return {"dauer": 9.0, "sprache": "german",
            "stats": {"A": {"dauer": 5.0, "turns": 1},
                      "B": {"dauer": 3.5, "turns": 1}},
            # Lang genug, um die Filterung zu überstehen — sonst zeigt das
            # Beispiel etwas, das es in der Ausgabe gar nicht mehr gibt.
            "overlaps": [{"start": 4.0, "ende": 6.0, "wer": ["A", "B"]}],
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
    festen = {"marta": "Martha"}
    text, geordnet = kern.glossar_schreiben(zaehler, ["Nils", "Anna"], festen)
    assert geordnet == ["Entwicklungsgespräch", "Kita"]
    gelesen, sprecher, gelesen_festen = kern.glossar_lesen(text)
    assert gelesen == zaehler and sprecher == ["Nils", "Anna"]
    # Bestätigte Korrekturen sind die einzige Wahrheit von außen — sie
    # dürfen beim Speichern nicht verlorengehen.
    assert gelesen_festen == festen


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
