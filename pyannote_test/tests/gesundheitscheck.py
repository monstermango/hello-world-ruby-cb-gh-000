"""Prüft, ob die App erreichbar und der Space betriebsbereit ist.

Startet den Space bei Bedarf neu. Läuft ohne HF-Token nur als Erreichbar-
keitsprüfung; mit Token (Umgebungsvariable HF_TOKEN) auch mit Neustart.

    python gesundheitscheck.py            # prüfen
    python gesundheitscheck.py --heilen   # prüfen und ggf. neu starten
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BACKEND = os.environ.get("BACKEND_SPACE", "Monstermango/diarization")
FRONTEND = os.environ.get(
    "FRONTEND_URL", "https://monstermango-sprecher-analyse.static.hf.space")
TOKEN = os.environ.get("HF_TOKEN", "")
# Zustände, aus denen sich der Space nicht von allein erholt
KAPUTT = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED"}


def _abrufen(url, methode="GET"):
    anfrage = urllib.request.Request(url, method=methode)
    if TOKEN:
        anfrage.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(anfrage, timeout=30) as antwort:
        return antwort.status, antwort.read()


def frontend_erreichbar():
    try:
        status, _ = _abrufen(FRONTEND + "/app.js")
        return status == 200
    except urllib.error.HTTPError as e:
        return e.code in (200, 302)
    except Exception:
        return False


def space_zustand():
    try:
        _, rohdaten = _abrufen(
            f"https://huggingface.co/api/spaces/{BACKEND}/runtime")
        return json.loads(rohdaten).get("stage", "UNBEKANNT")
    except Exception as e:
        return f"NICHT ABFRAGBAR ({type(e).__name__})"


def neu_starten():
    if not TOKEN:
        print("  kein HF_TOKEN — Neustart nicht möglich")
        return False
    try:
        _abrufen(f"https://huggingface.co/api/spaces/{BACKEND}/restart",
                 methode="POST")
        print("  Neustart angestoßen")
        return True
    except Exception as e:
        print(f"  Neustart fehlgeschlagen: {e}")
        return False


def main():
    heilen = "--heilen" in sys.argv
    zustand = space_zustand()
    frontend = frontend_erreichbar()
    print(f"Backend : {zustand}")
    print(f"Frontend: {'erreichbar' if frontend else 'NICHT erreichbar'}")

    gesund = zustand == "RUNNING" and frontend
    if gesund:
        print("Alles in Ordnung.")
        return 0

    if heilen and zustand in KAPUTT and neu_starten():
        for _ in range(20):                    # bis zu zehn Minuten warten
            time.sleep(30)
            if space_zustand() == "RUNNING":
                print("Backend wieder betriebsbereit.")
                return 0
        print("Backend nach Neustart weiterhin nicht bereit.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
