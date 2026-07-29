"""Fachlogik der Sprecher-Analyse — ohne Modelle, ohne Netz, ohne GPU.

Bewusst frei von schweren Abhängigkeiten: so lässt sich der gesamte
Kern in Sekunden und ohne Grafikkarte testen. `app.py` enthält nur noch
Modellaufrufe, Ablaufsteuerung und die API.
"""
import re
import unicodedata

import yaml

# Fensterung langer Aufnahmen
# Jeder GPU-Aufruf kostet Anstellzeit und Aufwärmen, unabhängig davon, wie
# viel Audio er verarbeitet. Große Fenster halbieren die Zahl der Aufrufe,
# ohne an der Erkennung selbst etwas zu ändern — die Fenster werden weiter
# einzeln und am Stück dekodiert. Nachgemessen tragen Reservierungen über
# 120 s (die Sonde lief mit 200 s durch), deshalb passt das Budget.
FENSTER = 600.0          # Audio je Transkriptionsfenster
FA_FENSTER = 240.0       # Audio je Alignment-Durchgang
FA_BLOCK = 1800.0        # Audio je HTTP-Anfrage beim Einpassen
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


def restzeit(verstrichen, getan, gesamt, dauer_audio=None):
    """Schätzt, wie lange der Auftrag noch braucht — in Sekunden.

    Sobald ein Abschnitt fertig ist, wird aus dessen Tempo hochgerechnet;
    das ist ehrlicher als jede Vorabformel, weil es die tatsächliche
    Warteschlange und Gerätelast enthält. Vorher bleibt nur eine grobe
    Schätzung aus der Audiolänge — als solche gekennzeichnet, damit
    niemand eine Genauigkeit annimmt, die es nicht gibt.
    """
    if gesamt <= 0 or getan >= gesamt:
        return None, "fertig"
    if getan > 0:
        pro = verstrichen / getan
        return max(0.0, pro * (gesamt - getan)), "gemessen"
    if dauer_audio:
        # Erfahrungswert der Kette: ungefähr ein Fünftel der Audiolänge,
        # plus Vorlauf für die Sprechererkennung.
        return max(0.0, dauer_audio * 0.2 + 30.0 - verstrichen), "geschätzt"
    return None, "unbekannt"


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

def _sprechpausen(turns):
    """Die echten Stillen: Lücken in der Vereinigung aller Redebeiträge.

    Ein Sprecherwechsel ist noch keine Pause — der nächste setzt oft ein,
    während der vorige noch redet. Erst die zusammengelegte Sprechzeit
    zeigt, wo tatsächlich niemand spricht.
    """
    if not turns:
        return []
    belegt = []
    for t in sorted(turns, key=lambda t: t["start"]):
        if belegt and t["start"] <= belegt[-1][1]:
            belegt[-1][1] = max(belegt[-1][1], t["ende"])
        else:
            belegt.append([t["start"], t["ende"]])
    return [(belegt[i][1], belegt[i + 1][0]) for i in range(len(belegt) - 1)
            if belegt[i + 1][0] > belegt[i][1]]


def _fenstergrenzen(turns, gesamt):
    """Schnittpunkte in die größte Sprechpause nahe der Zielmarke legen.

    Jeder Schnitt kostet Qualität: Whisper verliert am Fensteranfang und
    -ende Kontext, und ein mitten im Wort getrenntes Fenster erzeugt an
    beiden Seiten Unsinn. Landet der Schnitt dagegen in echter Stille,
    fängt das nächste Fenster sauber an. Deshalb die längste Pause im
    Suchbereich, nicht die nächstbeste Sprechergrenze.
    """
    pausen = _sprechpausen(turns)
    grenzen, pos = [0.0], 0.0
    while gesamt - pos > FENSTER:
        ziel = pos + FENSTER
        früh, spät = pos + 60, ziel + 120
        kandidaten = [((a + b) / 2, b - a) for a, b in pausen
                      if früh < (a + b) / 2 < spät]
        if kandidaten:
            # Längste Stille gewinnt; bei gleicher Länge die näher am Ziel.
            schnitt = max(kandidaten,
                          key=lambda p: (p[1], -abs(p[0] - ziel)))[0]
        else:
            # Keine Pause im Suchbereich: dann wenigstens an einem
            # Sprecherwechsel trennen statt blind auf der Marke.
            wechsel = [t["start"] for t in turns if früh < t["start"] < spät]
            schnitt = (min(wechsel, key=lambda x: abs(x - ziel))
                       if wechsel else ziel)
        grenzen.append(schnitt)
        pos = schnitt
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
            letztes["unsicher"].append(bool(wort.get("unsicher")))
        else:
            segmente.append({"start": wort["start"], "ende": wort["ende"],
                             "label": label, "worte": [wort["text"]],
                             "unsicher": [bool(wort.get("unsicher"))]})

    # Die Unsicherheit wird als Wortpositionen mitgeführt, nicht in den
    # Text gemischt: der Wortlaut muss unversehrt bleiben.
    fertig = [{"start": round(s["start"], 2), "ende": round(s["ende"], 2),
               "label": s["label"], "text": " ".join(s["worte"]),
               "unsicher": [i for i, u in enumerate(s["unsicher"]) if u]}
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

def _ausrichten(a, b):
    """Richtet zwei Wortfolgen aneinander aus.

    Gibt Paare (i, j) zurück; None steht für „hier fehlt etwas“. Basis für
    den Abgleich zweier Erkennungen und für die Fehlerrate — beide brauchen
    dieselbe Zuordnung, deshalb steht sie nur einmal hier.
    """
    zeilen = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        zeilen[i][0] = i
    for j in range(len(b) + 1):
        zeilen[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            zeilen[i][j] = min(zeilen[i - 1][j] + 1, zeilen[i][j - 1] + 1,
                               zeilen[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, paare = len(a), len(b), []
    while i > 0 or j > 0:
        if (i > 0 and j > 0
                and zeilen[i][j] == zeilen[i - 1][j - 1] + (a[i - 1] != b[j - 1])):
            paare.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and zeilen[i][j] == zeilen[i][j - 1] + 1:
            paare.append((None, j - 1))
            j -= 1
        else:
            paare.append((i - 1, None))
            i -= 1
    paare.reverse()
    return paare


def abgleich(vorlage, erkannt):
    """Wo zwei unabhängige Erkennungen auseinandergehen.

    Der stärkste kostenlose Hinweis auf Unsicherheit, den das System hat:
    Apple erkennt auf dem Gerät, Whisper auf der GPU — verschiedene
    Trainingsdaten, verschiedene Fehler. Wo beide dasselbe schreiben, ist
    es fast sicher richtig; wo sie sich widersprechen, lohnt das Nachhören.

    Der Wortlaut der Vorlage bleibt unangetastet. Zurückgegeben wird nur,
    an welchen Stellen sie bestritten wird und wodurch — markieren statt
    ersetzen, denn keine der beiden Quellen ist die Wahrheit.
    """
    v, e = _wortfolge(" ".join(vorlage)), _wortfolge(" ".join(erkannt))
    # Nur vergleichbar, wenn beide Folgen nach der Normalisierung noch zu
    # den Originalwörtern passen; sonst lieber gar nichts behaupten.
    if len(v) != len(vorlage) or not e:
        return []
    strittig = []
    for i, j in _ausrichten(v, e):
        if i is None:
            continue
        if j is None:
            strittig.append({"index": i, "wort": vorlage[i], "gehoert": None})
        elif v[i] != e[j]:
            strittig.append({"index": i, "wort": vorlage[i],
                             "gehoert": erkannt[j]})
    return strittig


def _wortfolge(text):
    """Text auf vergleichbare Wörter herunterbrechen.

    Für die Fehlerrate zählt der Wortlaut, nicht die Schreibweise:
    Groß-/Kleinschreibung und Satzzeichen fallen weg, Zahlen bleiben.
    """
    text = (text or "").lower().replace("ß", "ss")
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        text = text.replace(a, b)
    return re.findall(r"[a-z0-9]+", text)


def wortfehlerrate(referenz, hypothese):
    """Vergleicht zwei Transkripte Wort für Wort.

    Die übliche Maßzahl für Spracherkennung: wie viele Ersetzungen,
    Einfügungen und Löschungen nötig sind, um die Erkennung auf die
    Vorlage zu bringen, geteilt durch deren Länge. Ohne diese Zahl ist
    jede Aussage über „bessere Qualität“ eine Vermutung.
    """
    r, h = _wortfolge(referenz), _wortfolge(hypothese)
    ers, ein, loe, paare = 0, 0, 0, {}
    for i, j in _ausrichten(r, h):
        if i is None:
            ein += 1
        elif j is None:
            loe += 1
        elif r[i] != h[j]:
            ers += 1
            paare[(r[i], h[j])] = paare.get((r[i], h[j]), 0) + 1

    return {"woerter": len(r), "ersetzungen": ers, "einfuegungen": ein,
            "loeschungen": loe,
            "rate": (ers + ein + loe) / len(r) if r else 0.0,
            "haeufigste": sorted(paare.items(), key=lambda p: -p[1])[:15]}


def protokoll_text(markdown):
    """Zieht den gesprochenen Wortlaut aus einem erzeugten Markdown."""
    zeilen, drin = [], False
    for z in markdown.splitlines():
        if z.startswith("## "):
            drin = z.strip().lower() == "## protokoll"
            continue
        if drin and z.strip():
            zeilen.append(re.sub(r"^\*\*.*?\*\*\s*\([^)]*\):\s*", "", z.strip()))
    return " ".join(zeilen)


# ---------------------------------------------------------------- Konfidenz
# Der Aligner liefert für jedes Wort einen Anpassungswert. Bisher wurde er
# verworfen — damit sah ein wackliges Wort im Protokoll genauso sicher aus
# wie ein zweifelsfreies. Die Schwelle ist ein Startwert; mit
# tests/wortfehlerrate.py lässt sie sich an echtem Material nachziehen.
KONFIDENZ_ANTEIL = 0.10   # so viel vom Text darf höchstens wackelig sein
KONFIDENZ_DECKEL = 0.90   # darüber gilt ein Wort nie als unsicher


def _konfidenz_markieren(chunks, strittig_idx=(), anteil=KONFIDENZ_ANTEIL):
    """Markiert die Wörter, denen man am wenigsten trauen sollte.

    Bewusst relativ statt an einem festen Wert. Ein absoluter Schnitt
    setzt voraus, dass man die Werteverteilung des Aligners kennt — tut
    man nicht, und beim ersten echten Lauf lag praktisch der gesamte Text
    darunter. Alles zu markieren sagt genauso wenig wie nichts zu
    markieren.

    Also: das wackeligste Zehntel dieser Aufnahme, gemessen an ihrer
    eigenen Verteilung. Der Deckel verhindert, dass in einer rundum
    sauberen Aufnahme trotzdem Wörter angestrichen werden, nur weil
    irgendeines das schlechteste sein muss.

    Ein Widerspruch der zweiten Erkennung zählt unabhängig davon — der ist
    keine Frage des Grades.
    """
    strittig_idx = set(strittig_idx)
    werte = sorted(c["wert"] for c in chunks
                   if c.get("wert") is not None)
    grenze = None
    if werte:
        pos = max(0, min(len(werte) - 1, int(len(werte) * anteil) - 1))
        grenze = min(werte[pos], KONFIDENZ_DECKEL)

    for i, c in enumerate(chunks):
        wert = c.get("wert")
        c["unsicher"] = bool(
            i in strittig_idx
            or (wert is not None and grenze is not None and wert <= grenze))
    return chunks


# ------------------------------------------------------------ Stimmabdrücke
# Der Diarisierer berechnet für jeden Sprecher einen Stimmabdruck und wirft
# ihn nach dem Clustern weg. Dadurch begegnet die App denselben Personen
# jede Woche als Fremden. Aufgehoben erkennt sie sie wieder.
STIMME_SCHWELLE = 0.65    # darunter gilt es nicht als dieselbe Person
STIMME_ABSTAND = 0.06     # so viel Vorsprung braucht der beste Treffer
STIMME_TRAEGHEIT = 0.7    # so stark zählt der bisherige Abdruck nach


def _kosinus(a, b):
    if not a or not b or len(a) != len(b):
        return -1.0
    punkt = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return punkt / (na * nb) if na and nb else -1.0


def stimmen_zuordnen(abdruecke, bekannt):
    """Ordnet erkannte Sprecher bekannten Stimmen zu.

    Zurückhaltend: ein Treffer zählt nur, wenn er die Schwelle reißt *und*
    deutlich vor dem zweitbesten liegt. Zwei ähnliche Stimmen zu
    verwechseln wäre schlimmer, als beide unbenannt zu lassen — ein
    falscher Name im Protokoll ist schwer zu bemerken.
    """
    if not abdruecke or not bekannt:
        return {}
    treffer, vergeben = {}, set()
    kandidaten = []
    for label, vektor in abdruecke.items():
        werte = sorted(((_kosinus(vektor, v), name)
                        for name, v in bekannt.items()), reverse=True)
        if not werte:
            continue
        bester, name = werte[0]
        zweiter = werte[1][0] if len(werte) > 1 else -1.0
        if bester >= STIMME_SCHWELLE and bester - zweiter >= STIMME_ABSTAND:
            kandidaten.append((bester, label, name))
    for _, label, name in sorted(kandidaten, reverse=True):
        if name not in vergeben and label not in treffer:
            treffer[label] = name
            vergeben.add(name)
    return treffer


def stimmen_auffrischen(bekannt, abdruecke, namen):
    """Schreibt die Stimmabdrücke der benannten Sprecher fort.

    Gleitender Mittelwert statt Überschreiben: eine einzelne Aufnahme mit
    Schnupfen oder schlechtem Mikro soll die gespeicherte Stimme nicht
    umwerfen.
    """
    neu = {k: list(v) for k, v in (bekannt or {}).items()}
    for label, name in (namen or {}).items():
        vektor = (abdruecke or {}).get(label)
        if not vektor or not name:
            continue
        alt = neu.get(name)
        if alt and len(alt) == len(vektor):
            g = STIMME_TRAEGHEIT
            neu[name] = [g * a + (1 - g) * b for a, b in zip(alt, vektor)]
        else:
            neu[name] = list(vektor)
    return neu


# ------------------------------------------------------- Aufnahmequalität
# Was nicht im Signal steckt, holt kein Modell zurück. Diese Grenze war
# bisher unsichtbar — der Nutzer sah nur ein schlechtes Ergebnis und
# konnte nicht wissen, dass es an der Aufnahme lag.
SNR_GUT = 25.0
SNR_GRENZ = 15.0
LEISE_DB = -34.0
UEBERSTEUERT_ANTEIL = 0.005


def aufnahme_urteil(mess):
    """Beurteilt die Aufnahme und sagt, was beim nächsten Mal zu tun ist."""
    snr = (mess.get("sprache_db") or -60.0) - (mess.get("rausch_db") or -60.0)
    rat = []
    if snr < SNR_GRENZ:
        stufe = "schlecht"
        rat.append("Gerät näher an die Sprechenden legen — möglichst in die "
                   "Mitte und auf eine weiche Unterlage.")
    elif snr < SNR_GUT:
        stufe = "grenzwertig"
        rat.append("Etwas näher heran, dann wird der Wortlaut deutlich "
                   "zuverlässiger.")
    else:
        stufe = "gut"

    if (mess.get("sprache_db") or 0.0) < LEISE_DB:
        rat.append("Die Aufnahme ist sehr leise.")
    if (mess.get("uebersteuert") or 0.0) > UEBERSTEUERT_ANTEIL:
        stufe = "schlecht" if stufe == "grenzwertig" else stufe
        rat.append("Stellenweise übersteuert — nicht direkt vor dem Mikrofon "
                   "sprechen.")

    return {"stufe": stufe, "snr_db": round(snr, 1),
            "sprache_db": round(mess.get("sprache_db") or -60.0, 1),
            "rausch_db": round(mess.get("rausch_db") or -60.0, 1),
            "rat": rat}


# Halluzinationen in Stille
# Whisper ist zu erheblichen Teilen auf Untertiteln trainiert. Bekommt es
# ein Stück ohne Sprache, füllt es die Lücke mit dem, was in solchen
# Dateien am Ende steht — Abspann, Kanalhinweis, Dankesformel. Das sind
# keine Hörfehler, sondern erfundener Text, und er gehört nicht in ein
# Protokoll.
HALLUZINATIONEN = [
    re.compile(m, re.I) for m in (
        r"^untertitel(ung)?\b",
        r"\bamara\.org\b",
        r"\bim auftrag des zdf\b",
        r"\bfür funk,?\s*\d{4}",
        r"^vielen dank\.?$",
        r"\bdanke f(ü|ue)rs? (zuschauen|zusehen|zuh(ö|oe)ren)\b",
        r"\bbis zum n(ä|ae)chsten mal\b",
        r"\babonniert?\b.*\bkanal\b",
        r"\bmehr infos? (auf|unter)\b",
        r"^copyright\b", r"^\W*$",
    )
]
WIEDERHOLUNG_MAX = 4     # so oft darf dieselbe Wortfolge hintereinander stehen


def _ist_halluzination(text):
    """Erkennt erfundenen Untertitel-Text und festgefahrene Wiederholungen."""
    t = (text or "").strip()
    if not t:
        return True
    if any(m.search(t) for m in HALLUZINATIONEN):
        return True
    # Festgefahrene Dekodierung: dasselbe Wort immer wieder. Der
    # Temperatur-Rückfall fängt das meiste ab, aber nicht alles.
    worte = [w.lower() for w in re.findall(r"[\wÄÖÜäöüß-]+", t)]
    if len(worte) >= WIEDERHOLUNG_MAX * 2 and len(set(worte)) <= 2:
        return True
    lauf, letztes, laengster = 0, None, 0
    for w in worte:
        lauf = lauf + 1 if w == letztes else 1
        letztes = w
        laengster = max(laengster, lauf)
    return laengster > WIEDERHOLUNG_MAX


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


def _korrigieren(segmente, ziele, festen=None):
    """Rückt knapp danebenliegende Wörter auf bekannte Begriffe zurecht.

    Bewusst zurückhaltend. Korrigiert wird nur, was nah genug an genau
    einem Ziel liegt — passt ein Wort auf zwei Begriffe gleich gut, bleibt
    es unangetastet, denn dann ist die Korrektur geraten. Ebenso bleibt
    stehen, was selbst ein geläufiges Wort ist: „Wagen“ soll nicht zu
    „Hagen“ werden, nur weil jemand so heißt.

    Gibt die geänderten Segmente und ein Protokoll zurück — nichts wird
    still verändert.
    """
    festen = festen or {}
    if not ziele and not festen:
        return segmente, []

    entscheidung = {}   # Wort (klein) -> Ziel oder None
    protokoll = {}

    def ziel_fuer(wort):
        if wort in entscheidung:
            return entscheidung[wort]
        # Vom Nutzer bestätigte Korrekturen gelten ohne weitere Prüfung:
        # sie sind die einzige Wahrheit, die nicht aus der Erkennung
        # selbst stammt.
        if wort in festen:
            entscheidung[wort] = festen[wort]
            return festen[wort]
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
    zaehler, sprecher, festen, abschnitt = {}, [], {}, None
    for zeile in text.splitlines():
        z = zeile.strip()
        if z.lower().startswith("## sprecher"):
            abschnitt = "sprecher"
        elif z.lower().startswith("## korrekturen"):
            abschnitt = "korrekturen"
        elif z.startswith("## "):
            abschnitt = "begriffe"
        elif z.startswith("- ") and abschnitt:
            wert = z[2:].strip()
            if not wert:
                continue
            if abschnitt == "sprecher":
                if wert not in sprecher:
                    sprecher.append(wert)
            elif abschnitt == "korrekturen":
                if "->" in wert:
                    falsch, richtig = (t.strip() for t in wert.split("->", 1))
                    if falsch and richtig:
                        festen[falsch.lower()] = richtig
            else:
                treffer = re.match(r"^(.*?)\s*\((\d+)\)$", wert)
                begriff = (treffer.group(1) if treffer else wert).strip()
                anzahl = int(treffer.group(2)) if treffer else 1
                if begriff:
                    zaehler[begriff] = zaehler.get(begriff, 0) + anzahl
    return zaehler, sprecher, festen


def glossar_schreiben(zaehler, sprecher, festen=None):
    """Erzeugt die Glossardatei, häufigste Begriffe zuerst."""
    geordnet = _rangfolge(zaehler)[:GLOSSAR_MAX]
    zeilen = ["# Glossar", "",
              "Wird automatisch gepflegt und nach Häufigkeit sortiert; die",
              "obersten Einträge gehen als Vorwissen an die Erkennung.",
              "", "## Begriffe", ""]
    zeilen += [f"- {b} ({zaehler[b]})" for b in geordnet]
    zeilen += ["", "## Sprecher", ""]
    zeilen += [f"- {s}" for s in sprecher]
    # Vom Nutzer bestätigte Korrekturen. Das ist die einzige Wahrheit im
    # System, die nicht aus der Erkennung selbst stammt — sie gilt
    # unbedingt, nicht nur bei genügender Ähnlichkeit.
    zeilen += ["", "## Korrekturen", ""]
    zeilen += [f"- {f} -> {r}" for f, r in sorted((festen or {}).items())]
    return "\n".join(zeilen) + "\n", geordnet


def pfad_erlaubt(pfad):
    """Nur Aufnahmen aus dem eigenen Ordner dürfen gelesen werden."""
    return bool(
        isinstance(pfad, str)
        and re.fullmatch(r"aufnahmen/[A-Za-z0-9_.-]+\.md", pfad)
        and ".." not in pfad
    )
