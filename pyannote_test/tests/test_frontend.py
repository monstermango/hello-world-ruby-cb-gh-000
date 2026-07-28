"""E2E-Test der PWA im iPhone-Viewport mit gemocktem Backend/CDN."""
import asyncio
import http.server
import tempfile
import threading
import wave
from playwright.async_api import async_playwright

from pathlib import Path

WURZEL = str(Path(__file__).resolve().parents[1] / "hf-space-frontend")
PORT = 7911
BROWSER = "/opt/pw-browsers/chromium"

MOCK_CLIENT = """
export class Client {
  static async connect(url, optionen) { window._verbindungsOptionen = optionen; return new Client(); }
  submit(ep, daten) {
    const self = this;
    return (async function* () {
      yield {type: "status", stage: "generating",
             progress_data: [{desc: "Erkenne Sprecher …"}]};
      const res = await self.predict(ep, daten);
      yield {type: "data", data: res.data};
    })();
  }
  async predict(ep, daten) {
    const key = daten[0];
    if (key !== "test-key") return {data: [{ok: false, fehler: "Ungültiger Zugangsschlüssel."}]};
    if (ep === "/glossar") return {data: [{ok: true, begriffe: ["Kita"], sprecher: ["Nils"], zaehler: {Kita: 7}}]};
    if (ep === "/glossar_speichern") { window._glossar = daten; return {data: [{ok: true, begriffe: (daten[1]||"").split(/\\s*\\n\\s*/).filter(Boolean), sprecher: (daten[2]||"").split(/\\s*\\n\\s*/).filter(Boolean)}]}; }
    // Bewusst langsam: nur so faellt auf, wenn der Aufrufer die Antwort
    // nicht abwartet und den Token-Status zu frueh abliest.
    if (ep === "/starten") { window._diarNum = daten[2]; window._auftragStand = 0; return {data: [{ok: true, auftrag: "j1"}]}; }
    if (ep === "/auftrag") {
      window._auftragStand = (window._auftragStand || 0) + 1;
      // Erst laufend, dann fertig — so wird auch das Nachfragen geprüft.
      if (window._auftragStand < 2) {
        return {data: [{ok: true, stand: "laeuft", schritt: "Text erkennen", von: 1, bis: 3}]};
      }
      return {data: [{ok: true, stand: "fertig", ergebnis: window._ergebnis}]};
    }
    if (ep === "/push_schluessel") return {data: [{ok: true, schluessel: "BTestKey"}]};
    if (ep === "/push_anmelden") { window._pushAbo = daten[1]; return {data: [{ok: true, geraete: 1}]}; }
    if (ep === "/lernen") { window._gelernt = daten; return {data: [{ok: true, falsch: daten[1], richtig: daten[2]}]}; }
    if (ep === "/status") {
      await new Promise((f) => setTimeout(f, 400));
      return {data: [{ok: true, token: true}]};
    }
    if (ep === "/verlauf") return {data: [{ok: true, eintraege: [
      {pfad: "aufnahmen/a.md", titel: "Aufnahme 2026-07-25 07:04", datum: "2026-07-25T07:04:53+02:00", sprecher: 2, dauer_s: 23.4},
      {pfad: "aufnahmen/b.md", titel: "Aufnahme 2026-07-25 06:30", datum: "2026-07-25T06:30:27+02:00", sprecher: 3, dauer_s: 107}
    ]}]};
    if (ep === "/eintrag") return {data: [{ok: true,
      frontmatter: {titel: "Aufnahme 2026-07-25 07:04"},
      body: "# Aufnahme\\n\\n## Redeanteile\\n\\n| Sprecher | Anteil |\\n|---|---|\\n| Sprecher 1 | 58 % |",
      markdown: "---\\ntitel: x\\n---\\n# Aufnahme"}]};
    if (ep === "/vorbereiten") return {data: [{ok: true, id: "sid1", dauer: 23.4, modus: daten[2] ? "transkript" : "erkennung", woerter: (daten[2]||"").split(/\s+/).filter(Boolean).length}]};
    if (ep === "/diarisieren") { window._diarNum = daten[2]; return {data: [{ok: true, sprecher: 2, abschnitte: 3}]}; }
    if (ep === "/transkribieren") {
      window._abschnitte = (window._abschnitte || 0) + 1;
      return {data: [{ok: true, index: daten[2], abschnitte: 3, weiter: window._abschnitte < 3}]};
    }
    if (ep === "/speichern") {
      window._gespeicherteNamen = daten[2];
      return {data: [{ok: true, pfad: "aufnahmen/a.md",
                      markdown: "---\\ntitel: x\\n---\\n# Aufnahme"}]};
    }
    if (ep === "/abschliessen" || ep === "/analysieren") return {data: [window._ergebnis]};
    return {data: [{ok: false, fehler: "unbekannt"}]};
  }
}
window._ergebnis = {ok: true,
      zeitpunkt: "2026-07-25T07:04:53+02:00",
      quelle: "dialog.wav", sprache: "german",
      neue_begriffe: ["Entwicklungsgespräch", "Hedi"], sprecher_bekannt: ["Nils"],
      // Wiedererkannte Stimme und schwache Aufnahme — beides muss sichtbar sein.
      vorschlag: {B: "Martha"},
      qualitaet: {stufe: "schlecht", snr_db: 9.4, rat: ["Gerät näher an die Sprechenden legen."]},
      abdruecke: {A: [1, 0], B: [0, 1]},
      namen: {A: "Sprecher 1", B: "Sprecher 2"},
      farben: {A: "#4e79a7", B: "#f28e2b"},
      daten: {dauer: 23.4,
        segmente: [
          {start: 0, ende: 6.1, label: "A", text: "Guten Tag, hier spricht der erste Sprecher.", unsicher: [4]},
          {start: 7, ende: 12.7, label: "B", text: "Hallo, ich bin die zweite Sprecherin."},
          {start: 13.6, ende: 19.3, label: "A", text: "Natürlich, gerne."},
          {start: 20.2, ende: 23.1, label: "B", text: "Wunderbar!"}],
        korrekturen: [{vorher: "Marta", nachher: "Martha", anzahl: 2}],
        stats: {A: {dauer: 11.7, turns: 2}, B: {dauer: 8.5, turns: 2}},
        // Das Backend liefert nur noch zusammengefasste Passagen, keine
        // Einzelmessungen — der Mock muss dasselbe tun.
        overlaps: [{start: 12.4, ende: 15.0, wer: ["A", "B"], anzahl: 2, summe: 2.6}]}};
"""


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(
        *a, directory=WURZEL, **k)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler).serve_forever()


def _hoerprobe():
    """Winzige WAV-Datei; der Inhalt spielt keine Rolle, das Backend ist gemockt."""
    pfad = Path(tempfile.gettempdir()) / "sa_test.wav"
    if not pfad.exists():
        with wave.open(str(pfad), "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
            f.writeframes(b"\0\0" * 16000)
    return pfad


async def querscrollung(page):
    """Wie weit die Seite waagerecht scrollt — muss immer 0 bleiben.

    Ein einziges zu breites Element schiebt sonst die gesamte Seite mit,
    samt Kopfzeile: genau so ging die Überschrift schon einmal verloren.
    """
    return await page.evaluate(
        "Math.max(0, document.documentElement.scrollWidth"
        " - document.documentElement.clientWidth)")


async def main():
    hoerprobe = _hoerprobe()
    threading.Thread(target=serve, daemon=True).start()
    async with async_playwright() as p:
        start = {"executable_path": BROWSER} if Path(BROWSER).exists() else {}
        browser = await p.chromium.launch(**start)
        ctx = await browser.new_context(viewport={"width": 390, "height": 844},
                                        device_scale_factor=2, is_mobile=True,
                                        has_touch=True)
        page = await ctx.new_page()
        await page.route("**/cdn.jsdelivr.net/npm/@gradio/client/+esm",
                         lambda r: r.fulfill(content_type="text/javascript",
                                             body=MOCK_CLIENT))
        await page.route("**/cdn.jsdelivr.net/npm/marked@12/+esm",
                         lambda r: r.fulfill(
                             content_type="text/javascript",
                             body="export const marked = {parse: (t) => "
                                  "'<h2>MD</h2><table><tbody><tr><td>x</td>"
                                  "</tr></tbody></table>'};"))
        fehler = []
        page.on("pageerror", lambda e: fehler.append(str(e)))
        await page.goto(f"http://127.0.0.1:{PORT}/?hf=hf_testtoken", wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # 1. Login-Overlay sichtbar?
        assert await page.locator("#login").is_visible(), "Login-Overlay fehlt"
        await page.screenshot(path=str(Path(tempfile.gettempdir()) / "fe_login.png"))

        # 2. Falscher Schlüssel -> Fehlermeldung
        await page.fill("#key-input", "falsch")
        await page.click("#btn-login")
        await page.wait_for_timeout(800)
        text = await page.locator("#login-fehler").inner_text()
        assert "Ungültig" in text, f"Fehlermeldung fehlt: {text!r}"

        # 3. Richtiger Schlüssel -> Overlay verschwindet
        await page.fill("#key-input", "test-key")
        await page.click("#btn-login")
        await page.wait_for_timeout(800)
        assert not await page.locator("#login").is_visible(), "Overlay bleibt"

        # 3a. Die Rückmeldung muss den Token-Status kennen, nicht raten.
        #     Ohne await auf die Statusabfrage stand hier die Falschmeldung
        #     "ohne gültiges HF-Token", obwohl das Token einwandfrei war.
        meldung = await page.locator("#toast").inner_text()
        assert "aktiv" in meldung, f"Token-Rückmeldung falsch: {meldung!r}"
        abzeichen = await page.locator("#token-status").inner_text()
        assert abzeichen == "Konto-Kontingent", \
            f"Abzeichen falsch: {abzeichen!r}"

        # 3b. Einstellungen lassen sich öffnen und wieder schließen
        await page.locator("#btn-key").dispatch_event("click")
        await page.wait_for_timeout(300)
        assert await page.locator("#login").is_visible(), "Dialog öffnet nicht"
        assert await page.locator("#btn-schliessen").is_visible(), \
            "kein Schließen-Knopf"
        await page.locator("#btn-schliessen").dispatch_event("click")
        await page.wait_for_timeout(300)
        assert not await page.locator("#login").is_visible(), \
            "Dialog laesst sich nicht schliessen"
        # Auch neben das Blatt tippen schließt
        await page.locator("#btn-key").dispatch_event("click")
        await page.wait_for_timeout(300)
        await page.locator("#login").dispatch_event("click")
        await page.wait_for_timeout(300)
        assert not await page.locator("#login").is_visible(), \
            "Tippen neben das Blatt schliesst nicht"

        # 3b. Die Anleitung muss trennen, was nötig ist und was nur hilft:
        #     ohne Audiodatei laeuft gar nichts, das Apple-Transkript ist
        #     eine Zugabe.
        anleitung = await page.locator("#anleitung").inner_text()
        assert "nötig" in anleitung and "Bonus" in anleitung, \
            f"Anleitung kennzeichnet Pflicht/Bonus nicht: {anleitung!r}"
        marken = await page.locator("#anleitung .marke").count()
        assert marken == 2, f"Erwartet 2 Marken, gefunden {marken}"
        # Die Marke gehoert neben die Ueberschrift, nicht in eine eigene
        # Zeile — die Blockregel fuer .schritte li span darf nicht greifen.
        anzeige = await page.evaluate(
            "getComputedStyle(document.querySelector('#anleitung .marke'))"
            ".display")
        assert anzeige == "inline-block", f"Marke bricht um: {anzeige}"
        await page.screenshot(
            path=str(Path(tempfile.gettempdir()) / "fe_start.png"))

        # 4. Analyse mit Datei
        await page.set_input_files("#file-input", str(hoerprobe))
        await page.wait_for_timeout(500)
        assert await page.locator("#audio-panel").is_visible(), "Audio-Panel fehlt"

        # 4a. Die Sprecherzahl muss ohne Aufklappen erreichbar sein — die
        #     automatische Schätzung erkennt regelmäßig zu viele Sprecher,
        #     und hinter „Optionen“ hat das Feld niemand gefunden.
        assert await page.locator("#num-speakers").is_visible(), \
            "Sprecherzahl nicht ohne Aufklappen sichtbar"

        # Auch am Transkriptfeld selbst muss stehen, dass es optional ist.
        kopf = await page.locator("#transkript-box summary").inner_text()
        assert "Bonus" in kopf, f"Transkript nicht als Bonus markiert: {kopf!r}"
        await page.fill("#num-speakers", "3")

        await page.click("#btn-analyse")
        # Vor dem Auftrag darf nicht stehen, man könne die App schließen —
        # während des Hochladens bricht damit alles ab und es entsteht nie
        # ein Auftrag. Genau das war der Fehlschlag im echten Betrieb.
        vorher = await page.locator("#progress-hinweis").inner_text()
        assert "geöffnet lassen" in vorher, f"falscher Hinweis: {vorher!r}"
        await page.wait_for_timeout(12000)
        assert await page.locator(".sa-bubble").count() == 4, "Sprechblasen fehlen"
        assert "Deutsch" in await page.locator(".sa-meta").inner_text(), \
            "Sprache fehlt"
        opt = await page.evaluate("window._verbindungsOptionen")
        assert opt and opt.get("token") == "hf_testtoken", \
            f"Token nicht an den Client übergeben: {opt}"
        # Der Browser darf die Abschnitte nicht mehr selbst durchlaufen.
        abschnitte = await page.evaluate("window._abschnitte")
        assert not abschnitte, \
            f"Client treibt die Verarbeitung noch selbst: {abschnitte}"
        # Der Hinweis muss sagen, was die Stelle bedeutet — die frühere
        # Fassung kippte nur eine Wand aus Zeitstempeln auf den Schirm.
        warnung = await page.locator(".w-overlap").inner_text()
        assert "unsicher" in warnung, f"Hinweis ohne Aussage: {warnung!r}"
        assert len(warnung) < 200, f"Hinweis zu lang ({len(warnung)}): {warnung!r}"

        # 4a2. Unsichere Wörter müssen als solche erkennbar sein — sonst
        #      tritt ein wackliges Wort mit derselben Autorität auf wie
        #      ein zweifelsfreies.
        wackelig = page.locator(".wackelig")
        assert await wackelig.count() == 1, "unsicheres Wort nicht markiert"
        assert (await wackelig.first.inner_text()) == "der", \
            "falsche Wortposition markiert"
        # Der Wortlaut selbst bleibt unversehrt.
        blase = await page.locator(".sa-bubble").first.inner_text()
        assert blase == "Guten Tag, hier spricht der erste Sprecher.", blase

        # 4a3. Schwache Aufnahme wird benannt und erklärt.
        qual = await page.locator(".w-audio").inner_text()
        assert "Aufnahmequalität" in qual and "näher" in qual, qual

        # 4a4. Wiedererkannte Stimme ist vorausgefüllt.
        felder = page.locator(".namen-feld")
        assert await felder.nth(1).input_value() == "Martha", \
            "wiedererkannte Stimme nicht vorausgefüllt"
        assert await felder.first.input_value() == "", \
            "unbekannte Stimme darf nicht geraten werden"

        gewuenscht = await page.evaluate("window._diarNum")
        assert gewuenscht == 3, \
            f"Sprecherzahl nicht ans Backend durchgereicht: {gewuenscht}"

        # Die Verarbeitung läuft jetzt im Space; der Browser fragt nur nach.
        # Genau das muss geprüft sein — sonst hängt sie wieder am Telefon.
        auftrag = await page.evaluate("window._auftragStand")
        assert auftrag and auftrag >= 2, \
            f"Auftrag wurde nicht nachverfolgt: {auftrag}"
        # Und erst danach die Entwarnung.
        nachher = await page.evaluate(
            "document.querySelector('#progress-hinweis').textContent")
        assert "schließen" in nachher, f"keine Entwarnung: {nachher!r}"
        ueber = await querscrollung(page)
        assert ueber == 0, f"Ergebnis scrollt {ueber}px waagerecht"

        # 4a. Das Glossar lernt sichtbar mit
        gelernt = await page.locator(".gelernt").first.inner_text()
        assert "Entwicklungsgespräch" in gelernt and "Hedi" in gelernt, gelernt
        # Korrekturen muessen ausgewiesen sein, nicht still passieren.
        korr = await page.locator(".gelernt").nth(1).inner_text()
        assert "Marta" in korr and "Martha" in korr, \
            f"Korrektur nicht ausgewiesen: {korr!r}"

        # 4b. Sprecher benennen und speichern. Bewusst mit einem langen
        #     Namen: die Legende darf dadurch nicht über den Rand wachsen.
        #     Nach dem Speichern sind die Namensfelder weg, deshalb passiert
        #     beides in einem Durchgang.
        LANG = "Dr. Marta Hensen-Wittkowski"
        assert await page.locator(".namen-feld").count() == 2, "Namensfelder fehlen"
        await page.locator(".namen-feld").first.fill(LANG)
        await page.locator("#btn-speichern").dispatch_event("click")
        await page.wait_for_timeout(800)
        # 4b2. Ein angetipptes Wort berichtigen — der einzige Weg, auf dem
        #      Wissen von außen ins System gelangt.
        await page.evaluate(
            "window.prompt = () => 'die'")
        await page.locator(".wackelig").first.dispatch_event("click")
        await page.wait_for_timeout(600)
        gelernt = await page.evaluate("window._gelernt")
        assert gelernt and gelernt[1] == "der" and gelernt[2] == "die", \
            f"Korrektur nicht ans Backend gemeldet: {gelernt}"
        assert await page.locator(".berichtigt").count() == 1, \
            "berichtigtes Wort nicht als solches gekennzeichnet"

        gespeichert = await page.evaluate("window._gespeicherteNamen")
        # Der vorausgefüllte Name der wiedererkannten Stimme wird
        # mitgespeichert — genau dafür ist er da.
        assert gespeichert == {"A": LANG, "B": "Martha"}, \
            f"Namen falsch: {gespeichert}"
        assert await page.locator("#btn-md-laden").is_visible(), "MD-Button fehlt"
        assert LANG in await page.locator(".sa-legs").inner_text(), \
            "Umbenennung nicht übernommen"
        ueber = await querscrollung(page)
        assert ueber == 0, f"Lange Sprechernamen scrollen {ueber}px waagerecht"
        await page.screenshot(path=str(Path(tempfile.gettempdir()) / "fe_ergebnis.png"), full_page=True)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(300)

        # 5. Verlauf + Eintrag (dispatch_event umgeht ein Koordinaten-Artefakt
        # der Mobilemulation; elementFromPoint bestätigt die Klickbarkeit)
        await page.locator('button[data-view="verlauf"]').dispatch_event("click")
        await page.wait_for_timeout(800)
        assert await page.locator(".v-card").count() == 2, "Verlaufskarten fehlen"
        await page.screenshot(path=str(Path(tempfile.gettempdir()) / "fe_verlauf.png"), full_page=True)
        await page.locator(".v-card").first.dispatch_event("click")
        await page.wait_for_timeout(800)
        assert await page.locator("#eintrag-detail").is_visible(), "Detail fehlt"
        assert await page.locator("#eintrag-inhalt table").count() == 1, "MD-Tabelle fehlt"
        await page.screenshot(path=str(Path(tempfile.gettempdir()) / "fe_eintrag.png"), full_page=True)

        # 6. Direktlink-Login (?key=...)
        await page.evaluate("localStorage.clear()")
        await page.goto(f"http://127.0.0.1:{PORT}/?key=test-key",
                        wait_until="networkidle")
        await page.wait_for_timeout(800)
        assert not await page.locator("#login").is_visible(), "?key greift nicht"
        url = page.url
        assert "key=" not in url, f"Schlüssel bleibt in URL: {url}"

        assert not fehler, f"JS-Fehler: {fehler}"
        await browser.close()
        print("Alle Frontend-Tests bestanden")


if __name__ == "__main__":
    pass


asyncio.run(main())
