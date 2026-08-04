"""Vollständiger Durchlauf gegen das Live-System.

Zugangsdaten kommen ausschließlich aus der Umgebung — sie gehören nicht
in die Versionsverwaltung:

    APP_KEY=... HF_TOKEN=... python vollpruefung.py
"""
import os
import sys
import time

from gradio_client import Client, handle_file

K = os.environ.get("APP_KEY", "")
TOKEN = os.environ.get("HF_TOKEN", "")
BACKEND = os.environ.get("BACKEND_URL",
                         "https://monstermango-diarization.hf.space")
HOERPROBE = os.environ.get("HOERPROBE", "aufnahme.m4a")

if not K:
    sys.exit("APP_KEY fehlt — Zugangsschlüssel als Umgebungsvariable setzen.")
TEXT = ("Guten Tag, hier spricht der erste Sprecher. Ich erzähle etwas über das "
        "Wetter von heute. Hallo, ich bin die zweite Sprecherin. Das klingt ja "
        "sehr interessant, erzähl mir mehr davon. Natürlich, gerne. Morgen soll "
        "es sonnig werden und ziemlich warm für die Jahreszeit. Wunderbar, dann "
        "können wir ja einen Ausflug machen.")

ergebnisse = []


def pruefe(name, bedingung, zusatz=""):
    ergebnisse.append((name, bool(bedingung)))
    print(f"  {'OK  ' if bedingung else 'FEHL'} {name}" + (f"  {zusatz}" if zusatz else ""))


c = Client(BACKEND, hf_token=TOKEN or None)

print("\n— Zugriffsschutz —")
pruefe("falscher Schlüssel abgewiesen",
       c.predict("falsch", api_name="/status").get("ok") is False)
pruefe("leerer Schlüssel abgewiesen",
       c.predict("", api_name="/status").get("ok") is False)
pruefe("Token erkannt", c.predict(K, api_name="/status").get("token") is True)
pruefe("anonym erkannt",
       Client(BACKEND).predict(K, api_name="/status").get("token") is False)
for boeser in ["../glossar.md", "glossar.md", "aufnahmen/../x.md", "/etc/passwd"]:
    pruefe(f"Pfad abgewehrt: {boeser}",
           c.predict(K, boeser, api_name="/eintrag").get("ok") is False)

print("\n— Glossar —")
c.predict(K, "", "", api_name="/glossar_speichern")
g = c.predict(K, api_name="/glossar")
pruefe("Glossar geleert", g.get("ok") and not g.get("begriffe"))
g = c.predict(K, "Entwicklungsgespräch\nKita", "Nils", api_name="/glossar_speichern")
pruefe("Begriffe gespeichert", g.get("begriffe") == ["Entwicklungsgespräch", "Kita"])
pruefe("Sprecher gespeichert", g.get("sprecher") == ["Nils"])

print("\n— Erkennung (schrittweise) —")
t0 = time.time()
v = c.predict(K, handle_file(HOERPROBE), "", "", api_name="/vorbereiten")
pruefe("Vorbereiten", v.get("ok"), f"Dauer {v.get('dauer')}s, Modus {v.get('modus')}")
d = c.predict(K, v["id"], 0, "german", api_name="/diarisieren")
pruefe("Sprechererkennung", d.get("ok"), f"{d.get('sprecher')} Sprecher")
i = 0
while True:
    r = c.predict(K, v["id"], i, api_name="/transkribieren")
    if not r.get("ok") or not r.get("weiter"):
        break
    i += 1
pruefe("Transkription", r.get("ok"))
a = c.predict(K, v["id"], api_name="/abschliessen")
seg = a["daten"]["segmente"]
pruefe("Abschließen", a.get("ok"), f"{len(seg)} Segmente, {time.time()-t0:.0f}s")
pruefe("Text vorhanden", all(s["text"] for s in seg))
pruefe("Sprache erkannt", a.get("sprache") == "german")
pruefe("bekannte Sprecher gemeldet", "Nils" in (a.get("sprecher_bekannt") or []))
pruefe("Wiederholung ist folgenlos",
       c.predict(K, v["id"], 0, api_name="/transkribieren").get("ok"))

print("\n— Speichern —")
sp = c.predict(K, a, {"SPEAKER_00": "Nils", "SPEAKER_01": "Anna"},
               api_name="/speichern")
md = sp.get("markdown", "")
pruefe("gespeichert", sp.get("ok"), sp.get("pfad"))
pruefe("Namen übernommen", "Nils" in md and "Anna" in md)
pruefe("Frontmatter vollständig",
       all(x in md for x in ["titel:", "dauer_s:", "sprache:", "redeanteile_s:"]))
pruefe("kein Rohtranskript mehr", "## Rohtranskript" not in md)
pruefe("neue Namen gemerkt", "Anna" in (sp.get("neue_begriffe") or []))

print("\n— Eigenes Transkript —")
v2 = c.predict(K, handle_file(HOERPROBE), TEXT, "", api_name="/vorbereiten")
pruefe("Modus erkannt", v2.get("modus") == "transkript", f"{v2.get('woerter')} Wörter")
d2 = c.predict(K, v2["id"], 0, "german", api_name="/diarisieren")
i = 0
while True:
    r2 = c.predict(K, v2["id"], i, api_name="/transkribieren")
    if not r2.get("ok") or not r2.get("weiter"):
        break
    i += 1
a2 = c.predict(K, v2["id"], api_name="/abschliessen")
volltext = " ".join(s["text"] or "" for s in a2["daten"]["segmente"])
pruefe("Text wortgetreu übernommen", "sonnig werden" in volltext, volltext[:60])

print("\n— Verlauf —")
h = c.predict(K, api_name="/verlauf")
pruefe("Verlauf gelesen", h.get("ok"), f"{len(h.get('eintraege', []))} Einträge")
if h.get("eintraege"):
    e = c.predict(K, h["eintraege"][0]["pfad"], api_name="/eintrag")
    pruefe("Eintrag gelesen", e.get("ok") and e.get("frontmatter", {}).get("titel"))

print("\n— Sitzungsschutz —")
pruefe("unbekannte Sitzung abgewiesen",
       c.predict(K, "erfunden", api_name="/abschliessen").get("ok") is False)

fehler = [n for n, ok in ergebnisse if not ok]
print(f"\n{len(ergebnisse) - len(fehler)} von {len(ergebnisse)} Prüfungen bestanden")
if fehler:
    print("FEHLGESCHLAGEN:", ", ".join(fehler))
