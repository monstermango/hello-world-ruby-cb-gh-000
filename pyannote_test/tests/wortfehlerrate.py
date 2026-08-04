#!/usr/bin/env python3
"""Misst, wie gut die Erkennung wirklich ist.

Solange niemand eine Zahl hat, ist jede Aussage über bessere Qualität eine
Vermutung — und jede Änderung an der Erkennung ein Blindflug. Dieses
Skript liefert die Zahl: die Wortfehlerrate der eigenen Erkennung gegen
eine Vorlage, deren Wortlaut als richtig gilt.

Als Vorlage eignet sich das Transkript aus Apples Sprachmemos. Es ist
nicht fehlerfrei, aber unabhängig von dieser Kette entstanden, und darauf
kommt es an: gemessen wird die Veränderung zwischen zwei Läufen, nicht die
absolute Wahrheit.

So geht es:

  1. Aufnahme in der App verarbeiten, das Transkriptfeld dabei LEER
     lassen — sonst wird die Vorlage gemessen statt der Erkennung.
  2. Ergebnis als Markdown herunterladen.
  3. Transkript aus Sprachmemos in eine Textdatei kopieren.
  4.  python3 wortfehlerrate.py aufnahme.md sprachmemo.txt

Nach jeder Änderung an der Erkennung dasselbe wiederholen und die Raten
vergleichen. Sinkt sie, war die Änderung gut; steigt sie, zurückbauen.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hf-space"))

from kern import _frontmatter, protokoll_text, wortfehlerrate  # noqa: E402


def _lesen(pfad):
    text = Path(pfad).read_text(encoding="utf-8")
    # Ein erzeugtes Markdown enthält Tabellen und Kopfdaten; davon darf
    # nichts in den Vergleich geraten.
    return protokoll_text(text) if "## Protokoll" in text else text


def _modell(pfad):
    """Welches Modell den gemessenen Text erzeugt hat, laut Kopfdaten."""
    try:
        fm, _ = _frontmatter(Path(pfad).read_text(encoding="utf-8"))
    except OSError:
        return None
    return fm.get("modell")


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    hyp, ref = _lesen(argv[1]), _lesen(argv[2])
    if not ref.strip():
        print("Die Vorlage ist leer.")
        return 2

    e = wortfehlerrate(ref, hyp)
    modell = _modell(argv[1])
    # Zwei Raten ohne Modellnamen daneben sind zwei Zahlen ohne Aussage.
    print(f"\n  Modell       {modell or 'nicht vermerkt'}")
    print(f"  Vorlage      {e['woerter']} Wörter")
    print(f"  Fehlerrate   {e['rate']:.1%}")
    print(f"    ersetzt    {e['ersetzungen']}")
    print(f"    eingefügt  {e['einfuegungen']}")
    print(f"    fehlend    {e['loeschungen']}")
    if e["haeufigste"]:
        print("\n  Häufigste Verwechslungen (Vorlage -> Erkennung):")
        for (soll, ist), n in e["haeufigste"]:
            print(f"    {n:3d}x  {soll} -> {ist}")
    print("\n  Zum Vergleichen: dieselbe Aufnahme nach jeder Änderung "
          "erneut messen.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
