"""Fachlogik der Sprecher-Analyse — ohne Modelle, ohne Netz, ohne GPU.

Bewusst frei von schweren Abhängigkeiten: so lässt sich der gesamte
Kern in Sekunden und ohne Grafikkarte testen. `app.py` enthält nur noch
Modellaufrufe, Ablaufsteuerung und die API.
"""
import re
import unicodedata

import yaml

# Fensterung langer Aufnahmen
FENSTER = 300.0          # Audio je Transkriptionsfenster
FA_FENSTER = 240.0       # Audio je Alignment-Durchgang
FA_BLOCK = 900.0         # Audio je HTTP-Anfrage beim Einpassen
FA_ANTEIL = 0.6          # bewusst weniger Text anbieten, als das Fenster fasst

# Glossar
GLOSSAR_DATEI = "glossar.md"
GLOSSAR_MAX = 250        # so viele Begriffe werden behalten
GLOSSAR_PROMPT = 40      # so viele gehen als Vorwissen weiter

FARBEN = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
          "#b07aa1", "#76b7b2", "#edc948", "#9c755f"]

def _zeit(s):
    m, sec = divmod(int(round(s)), 60)
    return f"{m}:{sec:02d}"


def _namen_farben(labels):
    namen = {lb: f"Sprecher {i + 1}" for i, lb in enumerate(sorted(labels))}
    farben = {lb: FARBEN[i % len(FARBEN)] for i, lb in enumerate(sorted(labels))}
    return namen, farben

def _fenstergrenzen(turns, gesamt):
    """Schnittpunkte möglichst in Sprechpausen legen, damit kein Wort zerfällt."""
    grenzen, pos = [0.0], 0.0
    while gesamt - pos > FENSTER:
        ziel = pos + FENSTER
        kandidaten = [t["start"] for t in turns if pos + 60 < t["start"] < ziel + 120]
        grenzen.append(min(kandidaten, key=lambda x: abs(x - ziel))
                       if kandidaten else ziel)
        pos = grenzen[-1]
    grenzen.append(gesamt)
    return grenzen

def _sprecher_bei(turns, zeitpunkt):
    """Welcher Sprecher ist zu diesem Zeitpunkt aktiv?"""
    treffer, bester_abstand, ersatz = [], None, None
    for t in turns:
        if t["start"] <= zeitpunkt <= t["ende"]:
            treffer.append(t)
        else:
            abstand = min(abs(zeitpunkt - t["start"]), abs(zeitpunkt - t["ende"]))
            if bester_abstand is None or abstand < bester_abstand:
                bester_abstand, ersatz = abstand, t
    if treffer:
        # Bei Überlappung der längste Redebeitrag — meist die tragende Stimme
        return max(treffer, key=lambda t: t["ende"] - t["start"])["label"]
    return ersatz["label"] if ersatz else None

def _zusammenfuegen(turns, stats, overlaps, chunks, gesamt, sprache=None):
    """Baut Sprechersegmente aus den einzeln verorteten Wörtern."""
    sprache = sprache or next(
        (c["sprache"] for c in chunks if c.get("sprache")), None)

    segmente = []
    for wort in sorted(chunks, key=lambda c: c["start"]):
        label = _sprecher_bei(turns, (wort["start"] + wort["ende"]) / 2)
        if label is None:
            continue
        letztes = segmente[-1] if segmente else None
        if (letztes and letztes["label"] == label
                and wort["start"] - letztes["ende"] < 2.0):
            letztes["ende"] = wort["ende"]
            letztes["worte"].append(wort["text"])
        else:
            segmente.append({"start": wort["start"], "ende": wort["ende"],
                             "label": label, "worte": [wort["text"]]})

    fertig = [{"start": round(s["start"], 2), "ende": round(s["ende"], 2),
               "label": s["label"], "text": " ".join(s["worte"])}
              for s in segmente]

    # Wurde nichts erkannt, wenigstens die Sprecherstruktur zeigen
    if not fertig:
        fertig = [{"start": t["start"], "ende": t["ende"],
                   "label": t["label"], "text": None} for t in turns]

    return {"dauer": round(gesamt, 1), "segmente": fertig, "stats": stats,
            "overlaps": overlaps, "sprache": sprache}

def _markdown(daten, quelle, zeitpunkt, namen=None):
    std_namen, _ = _namen_farben(daten["stats"])
    namen = {**std_namen, **(namen or {})}
    frontmatter = {
        "titel": f"Aufnahme {zeitpunkt:%Y-%m-%d %H:%M}",
        "datum": zeitpunkt.isoformat(timespec="seconds"),
        "dauer_s": daten["dauer"],
        "sprecher": len(daten["stats"]),
        "sprache": daten.get("sprache"),
        "quelle": quelle,
        "redeanteile_s": {namen[lb]: st["dauer"]
                          for lb, st in daten["stats"].items()},
    }
    md = ["---", yaml.safe_dump(frontmatter, allow_unicode=True,
                                sort_keys=False).strip(), "---", "",
          f"# {frontmatter['titel']}", ""]

    gesamt = sum(st["dauer"] for st in daten["stats"].values()) or 1.0
    md += ["## Redeanteile", "",
           "| Sprecher | Sprechzeit | Anteil | Redebeiträge |",
           "|---|---|---|---|"]
    for lb, st in sorted(daten["stats"].items(),
                         key=lambda x: x[1]["dauer"], reverse=True):
        md.append(f"| {namen[lb]} | {_zeit(st['dauer'])} min "
                  f"| {100 * st['dauer'] / gesamt:.0f} % | {st['turns']} |")

    md += ["", "## Protokoll", ""]
    for s in daten["segmente"]:
        if s["text"]:
            md.append(f"**{namen[s['label']]}** "
                      f"({_zeit(s['start'])}–{_zeit(s['ende'])}): {s['text']}")
            md.append("")

    md += ["## Segmente", "", "| Start | Ende | Sprecher |", "|---|---|---|"]
    for s in daten["segmente"]:
        md.append(f"| {_zeit(s['start'])} | {_zeit(s['ende'])} | {namen[s['label']]} |")

    if daten["overlaps"]:
        md += ["", "## Gleichzeitiges Sprechen", "",
               "| Start | Ende | Beteiligte |", "|---|---|---|"]
        for o in daten["overlaps"]:
            wer = ", ".join(namen[lb] for lb in o["wer"])
            md.append(f"| {_zeit(o['start'])} | {_zeit(o['ende'])} | {wer} |")

    return "\n".join(md) + "\n"

def _frontmatter(text):
    if text.startswith("---"):
        try:
            ende = text.index("\n---", 3)
            return (yaml.safe_load(text[3:ende]) or {}), text[ende + 4:]
        except (ValueError, yaml.YAMLError):
            pass
    return {}, text

UMLAUTE = {"ä": "a", "ö": "o", "ü": "u", "ß": "ss", "á": "a", "à": "a",
           "é": "e", "è": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o",
           "ú": "u", "ñ": "n", "ç": "c"}

def _romanisieren(wort):
    w = wort.lower()
    for k, v in UMLAUTE.items():
        w = w.replace(k, v)
    w = unicodedata.normalize("NFD", w)
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z']", "", w)


HAEUFIG = {
    "Ich", "Du", "Er", "Sie", "Es", "Wir", "Ihr", "Der", "Die", "Das", "Den",
    "Dem", "Ein", "Eine", "Einen", "Einem", "Und", "Aber", "Oder", "Also",
    "Dann", "Doch", "Noch", "Nur", "Schon", "Ja", "Nein", "Genau", "Okay",
    "Was", "Wie", "Wer", "Wo", "Wann", "Warum", "Weil", "Wenn", "Dass",
    "Hier", "Da", "Dort", "Jetzt", "Mal", "Ganz", "Sehr", "Mehr", "Gut",
    "Herr", "Frau", "Vielen", "Dank", "Hallo", "Tag", "Zeit", "Sache",
}

def _rangfolge(zaehler):
    """Häufigstes zuerst, bei Gleichstand alphabetisch."""
    return [b for b, _ in sorted(zaehler.items(), key=lambda p: (-p[1], p[0]))]

def _begriffe_zaehlen(segmente):
    """Zählt wiederkehrende Eigennamen und Fachbegriffe im Transkript."""
    haeufigkeit = {}
    for s in segmente:
        for wort in re.findall(r"\b[A-ZÄÖÜ][\wÄÖÜäöüß-]{3,}\b", s.get("text") or ""):
            if wort in HAEUFIG:
                continue
            haeufigkeit[wort] = haeufigkeit.get(wort, 0) + 1
    # Einmalige Treffer sind meist Satzanfänge, nicht der Rede wert
    return {w: n for w, n in haeufigkeit.items() if n >= 2}

def _kontext_bauen(eigener, glossar, grenze=380):
    """Baut den Erkennungs-Hinweis; das Modell verarbeitet nur wenig Text.

    Die häufigsten Begriffe stehen vorn, damit bei knappem Platz das
    Wichtigste ankommt.
    """
    teile = []
    for wert in ([eigener] if eigener else []) + glossar["sprecher"] \
            + glossar["begriffe"][:GLOSSAR_PROMPT]:
        wert = wert.strip()
        if wert and wert not in teile:
            teile.append(wert)
    ergebnis = ""
    for teil in teile:
        kandidat = (ergebnis + ", " + teil) if ergebnis else teil
        if len(kandidat) > grenze:
            break
        ergebnis = kandidat
    return ergebnis


def glossar_lesen(text):
    """Liest die Glossardatei -> (Begriff->Anzahl, Sprechernamen)."""
    zaehler, sprecher, abschnitt = {}, [], None
    for zeile in text.splitlines():
        z = zeile.strip()
        if z.lower().startswith("## sprecher"):
            abschnitt = "sprecher"
        elif z.startswith("## "):
            abschnitt = "begriffe"
        elif z.startswith("- ") and abschnitt:
            wert = z[2:].strip()
            if not wert:
                continue
            if abschnitt == "sprecher":
                if wert not in sprecher:
                    sprecher.append(wert)
            else:
                treffer = re.match(r"^(.*?)\s*\((\d+)\)$", wert)
                begriff = (treffer.group(1) if treffer else wert).strip()
                anzahl = int(treffer.group(2)) if treffer else 1
                if begriff:
                    zaehler[begriff] = zaehler.get(begriff, 0) + anzahl
    return zaehler, sprecher


def glossar_schreiben(zaehler, sprecher):
    """Erzeugt die Glossardatei, häufigste Begriffe zuerst."""
    geordnet = _rangfolge(zaehler)[:GLOSSAR_MAX]
    zeilen = ["# Glossar", "",
              "Wird automatisch gepflegt und nach Häufigkeit sortiert; die",
              "obersten Einträge gehen als Vorwissen an die Erkennung.",
              "", "## Begriffe", ""]
    zeilen += [f"- {b} ({zaehler[b]})" for b in geordnet]
    zeilen += ["", "## Sprecher", ""]
    zeilen += [f"- {s}" for s in sprecher]
    return "\n".join(zeilen) + "\n", geordnet


def pfad_erlaubt(pfad):
    """Nur Aufnahmen aus dem eigenen Ordner dürfen gelesen werden."""
    return bool(
        isinstance(pfad, str)
        and re.fullmatch(r"aufnahmen/[A-Za-z0-9_.-]+\.md", pfad)
        and ".." not in pfad
    )
