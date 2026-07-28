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

# Dekodierung
# Die Temperaturleiter ist Whispers Sicherheitsnetz: erkennt die Bibliothek
# an Kompressionsrate oder Wahrscheinlichkeit, dass ein Abschnitt entgleist
# ist, verwirft sie ihn und würfelt ihn mit mehr Zufall neu. Die Schwellen
# sind die aus der Referenzimplementierung.
TEMPERATUREN = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
KOMPRESSION_MAX = 1.35    # darüber wiederholt sich der Abschnitt
LOGPROB_MIN = -1.0        # darunter ist das Modell zu unsicher
STILLE_SCHWELLE = 0.6     # darüber gilt der Abschnitt als Stille
STRAHLEN = 2              # Strahlsuche, solange die Temperatur 0 ist


def _decode_optionen(sprache=None):
    """Baut die Dekodier-Vorgaben für Whisper.

    Ohne Temperaturleiter hat `condition_on_prev_tokens` kein Gegengewicht:
    die Bibliothek löst den Vorlauf-Kontext erst oberhalb von Temperatur
    0.5, und dorthin gelangt sie nur über den Rückfall. Fehlt er, kann sich
    eine einmal begonnene Wiederholung über den Rest der Aufnahme
    fortschreiben. Beides gehört deshalb zusammen gesetzt — oder gar nicht.

    Die Strahlsuche gilt nur bei Temperatur 0; sobald gewürfelt wird,
    schaltet die Bibliothek selbst auf einen Strahl zurück.
    """
    gk = {
        "task": "transcribe",
        "num_beams": STRAHLEN,
        "condition_on_prev_tokens": True,
        "temperature": TEMPERATUREN,
        "compression_ratio_threshold": KOMPRESSION_MAX,
        "logprob_threshold": LOGPROB_MIN,
        "no_speech_threshold": STILLE_SCHWELLE,
    }
    if sprache:
        gk["language"] = sprache
    return gk


# Gleichzeitiges Sprechen
UEBERLAPP_MIN = 1.0       # kürzeres ist normales Dazwischenreden
UEBERLAPP_LUECKE = 5.0    # dichter beieinander gehört zu einer Passage
UEBERLAPP_PASSAGE = 2.0   # so viel muss zusammenkommen, um zu zählen

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

def _dichte_stellen(overlaps, mindest=UEBERLAPP_MIN, luecke=UEBERLAPP_LUECKE,
                    schwelle=UEBERLAPP_PASSAGE):
    """Fasst gleichzeitiges Sprechen zu den Passagen zusammen, die zählen.

    Drei Siebe hintereinander, weil eines nicht reicht. Kurzes
    Dazwischenreden — ein „mhm“, ein „ja, genau“ — steckt in jedem Gespräch
    zu Dutzenden und sagt nichts über den Wortlaut aus; das fällt an
    `mindest`. Was übrig bleibt, gehört inhaltlich zusammen, wenn es dicht
    aufeinander folgt, und wird über `luecke` zu einer Passage vereint.
    Und eine Passage ist erst dann eine Meldung wert, wenn sich darin
    insgesamt `schwelle` Sekunden Überlappung summieren — ein einzelner
    Sekundenblitzer kostet ein paar Wörter, eine dichte Folge macht einen
    ganzen Abschnitt unzuverlässig. Aus siebzig Rohmessungen werden so
    eine Handvoll Stellen, an denen sich Nachhören wirklich lohnt.
    """
    lang = sorted((o for o in overlaps if o["ende"] - o["start"] >= mindest),
                  key=lambda o: o["start"])
    passagen = []
    for o in lang:
        letzte = passagen[-1] if passagen else None
        if letzte and o["start"] - letzte["ende"] <= luecke:
            letzte["ende"] = max(letzte["ende"], o["ende"])
            letzte["wer"] = sorted(set(letzte["wer"]) | set(o["wer"]))
            letzte["anzahl"] += 1
            letzte["summe"] = round(letzte["summe"] + o["ende"] - o["start"], 2)
        else:
            passagen.append({"start": o["start"], "ende": o["ende"],
                             "wer": sorted(o["wer"]), "anzahl": 1,
                             "summe": round(o["ende"] - o["start"], 2)})
    return [p for p in passagen if p["summe"] >= schwelle]


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
            "overlaps": _dichte_stellen(overlaps), "sprache": sprache}

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

    if daten.get("korrekturen"):
        md += ["", "## Korrigierte Begriffe", "",
               "Gegen das Glossar zurechtgerückt. Der ursprüngliche "
               "Wortlaut steht hier, damit die Änderung nachvollziehbar "
               "bleibt.", "",
               "| Erkannt | Eingesetzt | Anzahl |", "|---|---|---|"]
        for k in daten["korrekturen"]:
            md.append(f"| {k['vorher']} | {k['nachher']} | {k['anzahl']} |")

    if daten["overlaps"]:
        md += ["", "## Gleichzeitiges Sprechen", "",
               "Stellen, an denen mehrere so lange gleichzeitig gesprochen "
               "haben, dass der Wortlaut dort unsicher ist.", "",
               "| Von | Bis | Beteiligte |", "|---|---|---|"]
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


# Deutsch schreibt jedes Substantiv groß, deshalb reicht „großgeschrieben“
# als Merkmal für einen Fachbegriff nicht annähernd aus. Ohne eine breite
# Sperrliste führen Allerweltswörter die Rangfolge an — gemessen standen
# „Ihren“ und „Augen“ vor „Martha“ und „Hedi“.
HAEUFIG = {
    # Pronomen, auch die großgeschriebene Höflichkeitsform
    "Ich", "Du", "Er", "Sie", "Es", "Wir", "Ihr", "Ihre", "Ihren", "Ihrem",
    "Ihres", "Ihrer", "Ihnen", "Mich", "Dich", "Uns", "Euch", "Mein",
    "Meine", "Meinen", "Dein", "Deine", "Sein", "Seine", "Unser", "Unsere",
    "Man", "Jemand", "Niemand", "Etwas", "Nichts", "Alles",
    # Artikel und Bestimmer
    "Der", "Die", "Das", "Den", "Dem", "Des", "Ein", "Eine", "Einen",
    "Einem", "Eines", "Einer", "Kein", "Keine", "Keinen", "Diese", "Dieser",
    "Dieses", "Diesem", "Diesen", "Alle", "Allen", "Jede", "Jeder", "Jedes",
    "Jeden", "Manche", "Solche", "Welche", "Welcher", "Welches", "Beide",
    # Bindewörter und Umstandswörter
    "Und", "Oder", "Aber", "Also", "Dann", "Doch", "Noch", "Nur", "Schon",
    "Ja", "Nein", "Genau", "Okay", "Was", "Wie", "Wer", "Wo", "Wann",
    "Warum", "Weil", "Wenn", "Dass", "Denn", "Damit", "Deshalb", "Darum",
    "Dadurch", "Trotzdem", "Außerdem", "Allerdings", "Vielleicht",
    "Natürlich", "Eigentlich", "Wirklich", "Einfach", "Immer", "Nie",
    "Oft", "Manchmal", "Wieder", "Erst", "Gerade", "Hier", "Da", "Dort",
    "Jetzt", "Heute", "Morgen", "Gestern", "Mal", "Ganz", "Sehr", "Mehr",
    "Weniger", "Gut", "Besser", "Schlecht", "Groß", "Klein", "Viel",
    "Wenig", "Zusammen", "Sozusagen", "Quasi", "Halt", "Eben",
    # Häufige Verbformen, meist am Satzanfang
    "Ist", "Sind", "War", "Waren", "Habe", "Hat", "Haben", "Hatte",
    "Hatten", "Kann", "Können", "Konnte", "Muss", "Müssen", "Musste",
    "Will", "Wollen", "Wollte", "Soll", "Sollen", "Sollte", "Darf",
    "Dürfen", "Werde", "Wird", "Werden", "Wurde", "Wurden", "Gibt",
    "Geht", "Kommt", "Macht", "Sagt", "Sagte", "Denke", "Denken",
    "Glaube", "Glauben", "Finde", "Finden", "Sehe", "Sehen", "Weiß",
    "Wissen", "Meinen", "Gesagt", "Gemacht", "Gehabt",
    # Allerweltsnomen
    "Herr", "Frau", "Vielen", "Dank", "Hallo", "Tag", "Zeit", "Sache",
    "Sachen", "Mensch", "Menschen", "Leute", "Kind", "Kinder", "Jahr",
    "Jahre", "Woche", "Wochen", "Monat", "Monate", "Stunde", "Stunden",
    "Minute", "Minuten", "Frage", "Fragen", "Antwort", "Punkt", "Punkte",
    "Beispiel", "Moment", "Ende", "Anfang", "Seite", "Teil", "Grund",
    "Gründe", "Fall", "Weg", "Art", "Weise", "Ding", "Dinge", "Idee",
    "Ideen", "Gruppe", "Gruppen", "Auge", "Augen", "Hand", "Hände",
    "Kopf", "Herz", "Haus", "Arbeit", "Schule", "Familie", "Eltern",
    "Mutter", "Vater", "Sohn", "Tochter", "Freund", "Freunde", "Problem",
    "Probleme", "Thema", "Themen", "Situation", "Erfahrung", "Bereich",
    "Möglichkeit", "Stelle", "Person", "Personen", "Leben", "Welt",
    "Land", "Stadt", "Platz", "Raum", "Zimmer", "Nummer", "Name", "Namen",
    "Wort", "Worte", "Wörter", "Satz", "Text", "Bild", "Bilder", "Film",
    "Musik", "Essen", "Wasser", "Geld", "Euro", "Prozent", "Endeffekt",
}

# Nachträgliche Korrektur gegen das Glossar
KORREKTUR_MIN_LAENGE = 4    # kürzere Begriffe haben zu viele Nachbarn
KORREKTUR_MIN_ZAEHLER = 5   # so oft muss ein Begriff bestätigt sein


def _abstand(a, b, grenze):
    """Levenshtein-Abstand, abgebrochen sobald `grenze` überschritten ist.

    Der Abbruch ist nicht nur Tempo: er hält den Rückgabewert klein und
    eindeutig — alles jenseits der Grenze ist ohnehin kein Treffer.
    """
    if abs(len(a) - len(b)) > grenze:
        return grenze + 1
    vorige = list(range(len(b) + 1))
    for i, za in enumerate(a, 1):
        aktuelle = [i]
        for j, zb in enumerate(b, 1):
            aktuelle.append(min(vorige[j] + 1, aktuelle[j - 1] + 1,
                                vorige[j - 1] + (za != zb)))
        if min(aktuelle) > grenze:
            return grenze + 1
        vorige = aktuelle
    return vorige[-1]


def _korrekturziele(glossar, eigener=""):
    """Begriffe, gegen die überhaupt korrigiert werden darf.

    Was der Nutzer selbst eingetragen hat — Sprechernamen, das Feld
    „Namen & Begriffe“ — zählt sofort. Automatisch Gesammeltes erst, wenn
    es sich über mehrere Aufnahmen bestätigt hat; sonst zementiert ein
    einmaliger Hörfehler sich selbst.
    """
    ziele, gesehen = [], set()

    def dazu(wert):
        wert = (wert or "").strip()
        if (len(wert) >= KORREKTUR_MIN_LAENGE and " " not in wert
                and wert not in HAEUFIG and wert.lower() not in gesehen):
            gesehen.add(wert.lower())
            ziele.append(wert)

    for teil in re.split(r"[,;]", eigener or ""):
        dazu(teil)
    for name in (glossar.get("sprecher") or []):
        dazu(name)
    for begriff, anzahl in (glossar.get("zaehler") or {}).items():
        if anzahl >= KORREKTUR_MIN_ZAEHLER:
            dazu(begriff)
    return ziele


def _korrigieren(segmente, ziele):
    """Rückt knapp danebenliegende Wörter auf bekannte Begriffe zurecht.

    Bewusst zurückhaltend. Korrigiert wird nur, was nah genug an genau
    einem Ziel liegt — passt ein Wort auf zwei Begriffe gleich gut, bleibt
    es unangetastet, denn dann ist die Korrektur geraten. Ebenso bleibt
    stehen, was selbst ein geläufiges Wort ist: „Wagen“ soll nicht zu
    „Hagen“ werden, nur weil jemand so heißt.

    Gibt die geänderten Segmente und ein Protokoll zurück — nichts wird
    still verändert.
    """
    if not ziele:
        return segmente, []

    entscheidung = {}   # Wort (klein) -> Ziel oder None
    protokoll = {}

    def ziel_fuer(wort):
        if wort in entscheidung:
            return entscheidung[wort]
        treffer = None
        if wort.capitalize() not in HAEUFIG and len(wort) >= KORREKTUR_MIN_LAENGE:
            for z in ziele:
                if z.lower() == wort:
                    treffer = None
                    break
                # Der Anfangsbuchstabe muss stimmen. Ohne diese Bedingung
                # liegt jedes beliebige Wort einen Tausch neben einem Namen
                # — „Wagen“ würde zu „Hagen“. Verhörer bei Eigennamen
                # betreffen fast immer die Mitte, nicht den Anfang.
                if z[:1].lower() != wort[:1]:
                    continue
                budget = 1 if len(z) < 8 else 2
                if _abstand(wort, z.lower(), budget) <= budget:
                    if treffer is not None and treffer != z:
                        treffer = None       # mehrdeutig: Finger weg
                        break
                    treffer = z
        entscheidung[wort] = treffer
        return treffer

    def ersetze(treffer):
        wort = treffer.group(0)
        z = ziel_fuer(wort.lower())
        if not z or z == wort:
            return wort
        protokoll[(wort, z)] = protokoll.get((wort, z), 0) + 1
        return z

    neu = []
    for s in segmente:
        text = s.get("text")
        if text:
            s = dict(s, text=re.sub(r"\b[\wÄÖÜäöüß-]+\b", ersetze, text))
        neu.append(s)

    liste = [{"vorher": v, "nachher": n, "anzahl": a}
             for (v, n), a in sorted(protokoll.items(), key=lambda p: -p[1])]
    return neu, liste

def _rangfolge(zaehler):
    """Häufigstes zuerst, bei Gleichstand alphabetisch."""
    return [b for b, _ in sorted(zaehler.items(), key=lambda p: (-p[1], p[0]))]

def _begriffe_zaehlen(segmente):
    """Zählt wiederkehrende Eigennamen und Fachbegriffe im Transkript.

    Neben der Sperrliste zählt die Satzstellung: ein Wort, das ausnahmslos
    am Satzanfang steht, ist großgeschrieben, weil dort jedes Wort
    großgeschrieben wird — als Substantiv ausgewiesen ist es damit nicht.
    Verlangt wird deshalb mindestens ein Vorkommen mitten im Satz.
    """
    haeufigkeit, mitten = {}, set()
    for s in segmente:
        text = s.get("text") or ""
        for treffer in re.finditer(r"\b[A-ZÄÖÜ][\wÄÖÜäöüß-]{3,}\b", text):
            wort = treffer.group(0)
            if wort in HAEUFIG:
                continue
            haeufigkeit[wort] = haeufigkeit.get(wort, 0) + 1
            vorher = text[:treffer.start()].rstrip()
            if vorher and vorher[-1] not in ".!?:":
                mitten.add(wort)
    # Einmalige Treffer sind meist Zufall, nicht der Rede wert
    return {w: n for w, n in haeufigkeit.items() if n >= 2 and w in mitten}

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
