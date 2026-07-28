const BACKEND = "https://monstermango-diarization.hf.space";

const $ = (sel) => document.querySelector(sel);
let client = null;
let schluessel = localStorage.getItem("sa_key") || "";
let hfToken = localStorage.getItem("sa_hf") || "";
let audioDatei = null;
let letztesMarkdown = null;
let letzteAnalyse = null;

const SPRACHEN = {
  german: "Deutsch", english: "Englisch", french: "Französisch",
  spanish: "Spanisch", italian: "Italienisch", turkish: "Türkisch",
  russian: "Russisch", arabic: "Arabisch", dutch: "Niederländisch",
  polish: "Polnisch", portuguese: "Portugiesisch", ukrainian: "Ukrainisch",
};
let recorder = null;
let recChunks = [];
let recTimer = null;
let wakeLock = null;

// ---------- Hilfen ----------

const zeit = (s) => {
  s = Math.round(s || 0);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

const esc = (t) => String(t ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(t._x);
  t._x = setTimeout(() => (t.hidden = true), 4000);
}

async function verbinde() {
  if (!client) {
    const { Client } = await import(
      "https://cdn.jsdelivr.net/npm/@gradio/client/+esm");
    // Der Client liest die Option als `token`; ältere Versionen als
    // `hf_token`. Beide setzen, damit das Token sicher als Bearer-Header
    // mitgeht — sonst läuft alles auf dem anonymen GPU-Kontingent.
    const optionen = hfToken ? { token: hfToken, hf_token: hfToken } : {};
    client = await Client.connect(BACKEND, optionen);
  }
  return client;
}

async function rufe(endpunkt, daten) {
  const c = await verbinde();
  const res = await c.predict(endpunkt, daten);
  return res.data[0];
}

// Wie rufe(), meldet aber Zwischenstände des Backends (lange Aufnahmen).
async function rufeMitFortschritt(endpunkt, daten, melde) {
  const c = await verbinde();
  const job = c.submit(endpunkt, daten);
  let ergebnis = null;
  for await (const nachricht of job) {
    if (nachricht.type === "data") {
      ergebnis = nachricht.data[0];
    } else if (nachricht.type === "status" && melde) {
      const stufe = (nachricht.progress_data || [])[0];
      if (stufe && stufe.desc) melde(stufe.desc);
      else if (nachricht.stage === "pending" && nachricht.queue) melde("In der Warteschlange …");
    }
  }
  if (ergebnis === null) throw new Error("Keine Antwort vom Server.");
  return ergebnis;
}

// ---------- Login ----------

async function loginPruefen(neuerKey) {
  const antwort = await rufe("/verlauf", [neuerKey]);
  if (!antwort.ok) throw new Error(antwort.fehler || "Anmeldung fehlgeschlagen");
  schluessel = neuerKey;
  localStorage.setItem("sa_key", neuerKey);
  $("#login").hidden = true;
  einstellungenSchliessbar();
  verlaufAnzeigen(antwort.eintraege);
  // Muss abgewartet werden: der Aufrufer meldet dem Nutzer gleich, ob das
  // Token greift. Ohne await läse er den Anfangswert und behauptete auch
  // bei einwandfreiem Token, es fehle eines.
  await tokenStatusPruefen();
  fortsetzenAnbieten();
}

// Fragt das Backend, ob die Anfragen mit HF-Token ankommen. Ohne Token läuft
// alles auf dem sehr kleinen anonymen GPU-Kontingent.
let tokenAktiv = null;

// Ein nicht eingetragenes Token ist etwas anderes als ein abgelehntes —
// im ersten Fall fehlt eine Eingabe, im zweiten stimmt der Wert nicht.
function tokenMeldung() {
  if (tokenAktiv) return "HF-Token aktiv — GPU läuft über dein Konto.";
  if (tokenAktiv === null) return "Verbindung zum Backend gestört.";
  if (!hfToken) return "Kein HF-Token hinterlegt — nur kleines anonymes "
    + "GPU-Kontingent.";
  return "HF-Token wurde nicht erkannt — bitte prüfen.";
}

async function tokenStatusPruefen() {
  const feld = $("#token-status");
  try {
    const s = await rufe("/status", [schluessel]);
    tokenAktiv = !!s.token;
  } catch (e) {
    tokenAktiv = null;
  }
  feld.hidden = tokenAktiv === null;
  feld.textContent = tokenAktiv ? "Konto-Kontingent"
    : hfToken ? "Token abgelehnt" : "ohne Token";
  feld.classList.toggle("warn", tokenAktiv === false);
  $("#login-status").textContent = tokenMeldung();
}

$("#btn-login").addEventListener("click", async () => {
  const wert = $("#key-input").value.trim();
  if (!wert) {
    $("#login-fehler").textContent = "Bitte den Zugangsschlüssel eingeben.";
    return;
  }
  // Leeres Feld darf ein bereits gespeichertes Token nicht löschen.
  const eingabe = $("#hf-input").value.trim();
  if (eingabe) {
    hfToken = eingabe;
    localStorage.setItem("sa_hf", hfToken);
  }
  client = null;
  $("#login-fehler").textContent = "";
  $("#btn-login").disabled = true;
  try {
    await loginPruefen(wert);
    toast(tokenMeldung());
  } catch (e) {
    $("#login-fehler").textContent = e.message;
  } finally {
    $("#btn-login").disabled = false;
  }
});

$("#btn-push-test").addEventListener("click", async () => {
  const feld = $("#push-status");
  // In Stufen, jede einzeln berichtet. Beim ersten Versuch verschluckte
  // ein hängender Versand alles und übrig blieb „An error occurred“ —
  // eine Meldung, mit der niemand etwas anfangen kann.
  feld.textContent = "Melde dieses Gerät an …";
  const anmeldung = await benachrichtigungAnbieten();
  if (anmeldung) {
    feld.textContent = `Anmeldung fehlgeschlagen: ${anmeldung}`;
    return;
  }
  try {
    const z = await rufe("/push_stand", [schluessel]);
    feld.textContent = `Bibliothek: ${z.bibliothek}, `
      + `Geräte: ${(z.geraete || []).join(", ") || "keins"}. Sende Probe …`;
  } catch (e) {
    feld.textContent = "Zustand nicht abrufbar: " + e.message;
    return;
  }
  try {
    const d = await rufe("/push_pruefen", [schluessel]);
    if (d.ok) {
      feld.textContent = `Gesendet an ${d.zugestellt} von ${d.geraete} `
        + `Gerät(en). Kommt jetzt keine Nachricht, liegt es an iOS, `
        + `nicht an der Einrichtung.`;
    } else {
      feld.textContent = `Fehlgeschlagen: ${d.fehler || ""} `
        + ((d.meldungen || []).join(" | ") || "");
    }
  } catch (e) {
    feld.textContent = "Versand ohne Antwort (" + e.message
      + ") — vermutlich Zeitüberschreitung beim Zustelldienst.";
  }
});

$("#key-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#btn-login").click();
});

// Der Dialog lässt sich nur schließen, wenn bereits ein Schlüssel
// hinterlegt ist — sonst stünde man vor einer unbenutzbaren App.
function einstellungenSchliessbar() {
  $("#btn-schliessen").hidden = !schluessel;
}

function einstellungenOeffnen() {
  $("#key-input").value = schluessel;
  $("#hf-input").value = hfToken;
  $("#login-fehler").textContent = "";
  einstellungenSchliessbar();
  $("#login").hidden = false;
}

function einstellungenSchliessen() {
  if (!schluessel) return;
  $("#login").hidden = true;
}

$("#btn-key").addEventListener("click", einstellungenOeffnen);
$("#btn-schliessen").addEventListener("click", einstellungenSchliessen);

// Tippen neben das Blatt schließt ebenfalls — auf dem Handy die
// naheliegendste Geste.
$("#login").addEventListener("click", (e) => {
  if (e.target === $("#login")) einstellungenSchliessen();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#login").hidden) einstellungenSchliessen();
});

// ---------- Navigation ----------

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) =>
      b.classList.toggle("active", b === btn));
    const ziel = btn.dataset.view;
    $("#view-analyse").hidden = ziel !== "analyse";
    $("#view-verlauf").hidden = ziel !== "verlauf";
    if (ziel === "verlauf") verlaufLaden();
  });
});

// ---------- Aufnahme & Datei ----------

function setzeAudio(datei) {
  audioDatei = datei;
  const mb = datei.size / 1048576;
  $("#audio-name").textContent =
    `${datei.name} (${mb < 1 ? Math.round(datei.size / 1024) + " KB"
                             : mb.toFixed(1) + " MB"})`;
  $("#audio-preview").src = URL.createObjectURL(datei);
  $("#audio-panel").hidden = false;
  $("#anleitung").hidden = true;
  $("#ergebnis").innerHTML = "";
  letzteAnalyse = null;
}

$("#btn-datei").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => {
  const datei = e.target.files[0];
  if (!datei) return;
  if (!datei.size) {
    toast("Datei ist leer — liegt sie noch in iCloud? Erst herunterladen.");
    return;
  }
  setzeAudio(datei);
});
$("#btn-verwerfen").addEventListener("click", () => {
  audioDatei = null;
  $("#audio-panel").hidden = true;
  $("#anleitung").hidden = false;
  $("#ergebnis").innerHTML = "";
});

$("#btn-record").addEventListener("click", async () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported("audio/mp4")
      ? "audio/mp4" : "audio/webm";
    // Niedrige Bitrate: eine Stunde Sprache bleibt so bei ~20 MB.
    recorder = new MediaRecorder(stream, { mimeType: mime,
                                           audioBitsPerSecond: 48000 });
    recChunks = [];
    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size) recChunks.push(e.data);
    };
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recTimer);
      wakeLockFreigeben();
      $("#btn-record").classList.remove("aktiv");
      $("#btn-record").textContent = "oder direkt hier aufnehmen";
      $("#rec-status").hidden = true;
      $("#rec-status").classList.remove("laeuft");
      const endung = mime === "audio/mp4" ? "m4a" : "webm";
      setzeAudio(new File(recChunks, `aufnahme.${endung}`, { type: mime }));
    };
    // Sekündlich Daten abholen, damit lange Aufnahmen nicht am Stück im
    // Arbeitsspeicher gehalten werden.
    recorder.start(5000);
    await wakeLockAnfordern();
    $("#btn-record").classList.add("aktiv");
    $("#btn-record").textContent = "Aufnahme stoppen";
    $("#rec-status").hidden = false;
    $("#rec-status").classList.add("laeuft");
    const startZeit = Date.now();
    recTimer = setInterval(() => {
      const s = (Date.now() - startZeit) / 1000;
      $("#rec-status").textContent = `Aufnahme läuft … ${zeit(s)}`;
      if (s > 3600 && recorder.state === "recording") {
        recorder.stop();
        toast("Aufnahme nach 60 Minuten automatisch beendet.");
      }
    }, 500);
  } catch (e) {
    toast("Kein Mikrofonzugriff: " + e.message);
  }
});

// Bildschirmsperre verhindern — sonst pausiert iOS die Aufnahme.
async function wakeLockAnfordern() {
  try {
    if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen");
  } catch (e) { /* nicht verfügbar */ }
}

function wakeLockFreigeben() {
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
}

document.addEventListener("visibilitychange", async () => {
  const laeuft = (recorder && recorder.state === "recording")
              || !$("#progress").hidden;
  if (document.visibilityState === "visible" && laeuft && !wakeLock) {
    await wakeLockAnfordern();
  }
});

// ---------- Analyse ----------

// Wiederholt einen Schritt, wenn die Verbindung abbricht — etwa weil iOS die
// Seite beim Verdunkeln des Bildschirms eingefroren hat.
async function rufeHartnaeckig(endpunkt, daten, versuche = 3) {
  let letzter;
  for (let v = 0; v < versuche; v++) {
    try {
      return await rufe(endpunkt, daten);
    } catch (e) {
      letzter = e;
      client = null;
      await new Promise((r) => setTimeout(r, 1500 * (v + 1)));
    }
  }
  throw letzter;
}

function laufSpeichern(zustand) {
  localStorage.setItem("sa_lauf", JSON.stringify({ ...zustand, ts: Date.now() }));
}

function laufLoeschen() {
  localStorage.removeItem("sa_lauf");
  $("#fortsetzen").hidden = true;
}

async function abschnitteVerarbeiten(id, abschnitte, eigenerText, ab, melde) {
  for (let i = ab; ; i++) {
    const was = eigenerText ? "Ordne Text zu" : "Transkribiere";
    melde(abschnitte > 1
      ? `${was} — Abschnitt ${Math.min(i + 1, abschnitte)} von ${abschnitte} …`
      : `${was} …`);
    const tr = await rufeHartnaeckig("/transkribieren", [schluessel, id, i]);
    if (!tr.ok) throw new Error(tr.fehler);
    laufSpeichern({ id, i: i + 1, abschnitte, eigenerText });
    if (!tr.weiter) break;
  }
}

async function analyseRahmen(arbeit) {
  $("#btn-analyse").disabled = true;
  $("#fortsetzen").hidden = true;
  $("#progress").hidden = false;
  $("#ergebnis").innerHTML = "";
  const start = Date.now();
  const ticker = setInterval(() => {
    $("#progress-zeit").textContent = zeit((Date.now() - start) / 1000);
  }, 1000);
  await wakeLockAnfordern();          // Bildschirm an lassen
  const melde = (t) => ($("#progress-text").textContent = t);
  try {
    const d = await arbeit(melde);
    if (!d.ok) throw new Error(d.fehler);
    laufLoeschen();
    letzteAnalyse = d;
    ergebnisAnzeigen(d, false);
    // Ergebnis in den Blick holen — sonst steht es unter der Eingabe.
    // Etwas Vorlauf, damit die Kopfzeile nicht unter der Titelleiste liegt.
    window.scrollTo({ top: Math.max(0, $("#ergebnis").offsetTop - 88),
                      behavior: "smooth" });
  } catch (e) {
    let hinweis = "";
    if (/ZeroGPU|quota|runs limit/i.test(e.message) && tokenAktiv === false) {
      hinweis = " — Es ist kein HF-Token hinterlegt. Über ⚙︎ eintragen, dann "
              + "läuft die GPU über dein Konto.";
    }
    toast("Fehler: " + e.message + hinweis);
    fortsetzenAnbieten();
  } finally {
    clearInterval(ticker);
    wakeLockFreigeben();
    $("#btn-analyse").disabled = false;
    $("#progress").hidden = true;
  }
}

$("#btn-analyse").addEventListener("click", () => {
  if (!audioDatei) return;
  analyseRahmen(async (melde) => {
    const n = parseInt($("#num-speakers").value, 10) || 0;

    melde("Lade Audio hoch …");
    const transkript = $("#transkript").value.trim();
    const kontext = $("#kontext").value.trim();
    localStorage.setItem("sa_kontext", kontext);
    const vor = await rufeHartnaeckig(
      "/vorbereiten", [schluessel, audioDatei, transkript, kontext]);
    if (!vor.ok) throw new Error(vor.fehler);

    const spr = $("#sprache").value;
    localStorage.setItem("sa_sprache", spr);

    // Ab hier läuft die Arbeit im Space weiter, auch wenn das Telefon
    // schläft oder die App geschlossen wird. Der Browser fragt nur noch
    // nach dem Stand.
    const start = await rufeHartnaeckig("/starten",
                                        [schluessel, vor.id, n, spr]);
    if (!start.ok) throw new Error(start.fehler);
    // Ohne Kennung an der Startanfrage läuft die GPU-Zeit nicht über dein
    // Konto. Das soll auffallen, bevor die Aufnahme durchgelaufen ist.
    if (start.mit_token === false && hfToken) {
      toast("Achtung: Auftrag ohne dein HF-Token gestartet.");
    }
    auftragSpeichern(start.auftrag);
    benachrichtigungAnbieten();
    return await auftragVerfolgen(start.auftrag, melde);
  });
});

// ---------- Aufträge ----------

function auftragSpeichern(jid) {
  localStorage.setItem("sa_auftrag", JSON.stringify({ id: jid, ts: Date.now() }));
}

function auftragVergessen() {
  localStorage.removeItem("sa_auftrag");
}

async function auftragVerfolgen(jid, melde) {
  // Reines Nachfragen — bricht die Verbindung ab, läuft die Verarbeitung
  // trotzdem weiter und wird beim nächsten Öffnen wieder eingesammelt.
  for (;;) {
    let j;
    try {
      j = await rufe("/auftrag", [schluessel, jid]);
    } catch (e) {
      await new Promise((f) => setTimeout(f, 5000));
      continue;
    }
    if (!j.ok) throw new Error(j.fehler);
    if (j.stand === "fertig") {
      auftragVergessen();
      return j.ergebnis;
    }
    if (j.stand === "fehler") {
      auftragVergessen();
      throw new Error(j.fehler || "Verarbeitung fehlgeschlagen");
    }
    const von = j.von || 0, bis = j.bis || 1;
    melde(bis > 1 ? `${j.schritt} … ${von}/${bis}` : `${j.schritt} …`);
    await new Promise((f) => setTimeout(f, 4000));
  }
}

function laufenderAuftrag() {
  try {
    const a = JSON.parse(localStorage.getItem("sa_auftrag") || "null");
    if (a && a.id && Date.now() - a.ts < 6 * 3600 * 1000) return a.id;
  } catch (e) { /* verworfen */ }
  return null;
}

// ---------- Benachrichtigung ----------

async function benachrichtigungAnbieten() {
  // Auf dem iPhone geht Web Push nur in der zum Home-Bildschirm
  // hinzugefügten App, und die Erlaubnis muss aus einer Nutzergeste
  // kommen. Der Start der Analyse ist genau so eine.
  // Gibt den Grund zurück, wenn es nicht klappt — sonst bleibt jeder
  // Fehlschlag unsichtbar. Leerer Rückgabewert heißt: eingerichtet.
  if (!("Notification" in window) || !("serviceWorker" in navigator)) {
    return "Dieser Browser kann keine Benachrichtigungen. Die App muss vom "
      + "Home-Bildschirm geöffnet sein, nicht aus Safari.";
  }
  if (Notification.permission === "denied") {
    return "In den iOS-Einstellungen abgelehnt.";
  }
  try {
    if (Notification.permission === "default") {
      if (await Notification.requestPermission() !== "granted") {
        return "Erlaubnis nicht erteilt.";
      }
    }
    const reg = await navigator.serviceWorker.ready;
    let abo = await reg.pushManager.getSubscription();
    if (!abo) {
      const k = await rufe("/push_schluessel", [schluessel]);
      if (!k.ok) return "Kein Push-Schlüssel vom Backend.";
      abo = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: k.schluessel,
      });
    }
    const a = await rufe("/push_anmelden", [schluessel, abo.toJSON()]);
    if (!a.ok) return a.fehler || "Anmeldung abgelehnt.";
    return "";
  } catch (e) {
    // Benachrichtigungen sind Zugabe — ihr Fehlschlag darf die Analyse
    // nicht aufhalten. Gemeldet wird er trotzdem.
    console.warn("Push nicht eingerichtet:", e);
    return e.message || String(e);
  }
}

function fortsetzenAnbieten() {
  // Läuft im Space noch ein Auftrag, wird er beim Öffnen wieder
  // eingesammelt — auch wenn die App zwischendurch geschlossen war.
  if (laufenderAuftrag()) {
    $("#fortsetzen").hidden = false;
    $("#fortsetzen-text").textContent =
      "Eine Analyse läuft im Hintergrund weiter.";
    $("#btn-fortsetzen").textContent = "Stand ansehen";
    return;
  }
  let lauf;
  try {
    lauf = JSON.parse(localStorage.getItem("sa_lauf") || "null");
  } catch (e) {
    lauf = null;
  }
  if (!lauf || !lauf.id || Date.now() - lauf.ts > 2.5 * 3600 * 1000) return;
  $("#fortsetzen").hidden = false;
}

$("#btn-fortsetzen").addEventListener("click", () => {
  const jid = laufenderAuftrag();
  if (jid) {
    analyseRahmen((melde) => auftragVerfolgen(jid, melde));
    return;
  }
  const lauf = JSON.parse(localStorage.getItem("sa_lauf") || "null");
  if (!lauf) return;
  analyseRahmen(async (melde) => {
    if (lauf.abschnitte === null) {
      melde("Erkenne Sprecher …");
      const n = parseInt($("#num-speakers").value, 10) || 0;
      const dia = await rufeHartnaeckig(
        "/diarisieren", [schluessel, lauf.id, n, $("#sprache").value]);
      if (!dia.ok) throw new Error(dia.fehler);
      lauf.abschnitte = dia.abschnitte;
      lauf.i = 0;
    }
    await abschnitteVerarbeiten(lauf.id, lauf.abschnitte, lauf.eigenerText,
                                lauf.i, melde);
    melde("Stelle Ergebnis zusammen …");
    return await rufeHartnaeckig("/abschliessen", [schluessel, lauf.id]);
  });
});

function ergebnisAnzeigen(d, gespeichert) {
  const { daten, namen, farben } = d;
  const labels = Object.keys(namen).sort();
  const gesamt = labels.reduce((a, lb) => a + daten.stats[lb].dauer, 0) || 1;
  const dauer = daten.dauer || 1;
  const wann = new Date(d.zeitpunkt);
  const chip = (lb) =>
    `<span class="chip" style="background:${farben[lb]}">${esc(namen[lb])}</span>`;

  const spracheRoh = d.sprache || daten.sprache;
  const sprache = spracheRoh
    ? `<span>🌐 ${esc(SPRACHEN[spracheRoh] || spracheRoh)}</span>` : "";
  const meta =
    `<div class="sa-meta"><span>🕒 ${zeit(daten.dauer)} min</span>` +
    `<span>👥 ${labels.length} Sprecher</span>${sprache}` +
    `<span>📅 ${wann.toLocaleString("de-DE", { dateStyle: "short", timeStyle: "short" })}</span></div>`;

  const sortiert = [...labels].sort(
    (a, b) => daten.stats[b].dauer - daten.stats[a].dauer);
  let balken = "", legende = "";
  for (const lb of sortiert) {
    const anteil = (100 * daten.stats[lb].dauer) / gesamt;
    balken += `<div style="width:${anteil.toFixed(1)}%;background:${farben[lb]}"></div>`;
    legende += `<span class="sa-leg">${chip(lb)} <span class="dim">` +
      `${anteil.toFixed(0)} % · ${zeit(daten.stats[lb].dauer)} min</span></span>`;
  }

  let lanes = "";
  for (const lb of labels) {
    let bloecke = "";
    for (const s of daten.segmente) {
      if (s.label !== lb) continue;
      const links = (100 * s.start) / dauer;
      const breite = Math.max(0.5, (100 * (s.ende - s.start)) / dauer);
      bloecke += `<div style="left:${links.toFixed(2)}%;width:${breite.toFixed(2)}%;` +
        `background:${farben[lb]}"></div>`;
    }
    lanes += `<div class="sa-lane">${bloecke}</div>`;
  }
  const zeitleiste = `<div class="sa-timeline">${lanes}` +
    `<div class="sa-ticks"><span>0:00</span><span>${zeit(dauer)}</span></div></div>`;

  // Unsichere Wörter werden gekennzeichnet statt stillschweigend so
  // sicher dargestellt wie der Rest. Antippen berichtigt sie — das ist
  // zugleich der einzige Weg, auf dem echtes Wissen ins System kommt.
  const wortlaut = (s) => {
    const unsicher = new Set(s.unsicher || []);
    if (!unsicher.size) return esc(s.text);
    return s.text.split(/\s+/).map((w, i) => unsicher.has(i)
      ? `<span class="wackelig" data-wort="${esc(w)}" role="button"
              tabindex="0" title="Antippen zum Berichtigen">${esc(w)}</span>`
      : esc(w)).join(" ");
  };

  let gespraech = `<div class="sa-h">Gespräch</div>`;
  for (const s of daten.segmente) {
    if (!s.text) continue;
    gespraech +=
      `<div class="sa-msg"><div class="sa-rail" style="background:${farben[s.label]}"></div>` +
      `<div class="sa-msg-body"><div class="sa-msg-head">${esc(namen[s.label])} · ` +
      `${zeit(s.start)}–${zeit(s.ende)}</div>` +
      `<div class="sa-bubble">${wortlaut(s)}</div></div></div>`;
  }

  // Das Glossar pflegt sich selbst — hier wird nur nachvollziehbar, dass
  // und was es gelernt hat. Bedienen muss man dafür nichts.
  let gelernt = "";
  if ((d.neue_begriffe || []).length) {
    const woerter = d.neue_begriffe.slice(0, 6).map(esc).join(", ");
    const rest = d.neue_begriffe.length > 6
      ? ` und ${d.neue_begriffe.length - 6} weitere` : "";
    gelernt = `<div class="gelernt">📖 Neu gemerkt: ${woerter}${rest}. ` +
      `Künftige Aufnahmen werden dadurch treffsicherer.</div>`;
  }

  // Korrekturen werden ausgewiesen, nicht stillschweigend vorgenommen —
  // wer den Wortlaut prüfen will, muss sehen, was verändert wurde.
  if ((daten.korrekturen || []).length) {
    const k = daten.korrekturen.slice(0, 6)
      .map((x) => `${esc(x.vorher)} → ${esc(x.nachher)}`).join(", ");
    const rest = daten.korrekturen.length > 6
      ? ` und ${daten.korrekturen.length - 6} weitere` : "";
    gelernt += `<div class="gelernt">✏️ Gegen das Glossar korrigiert: ` +
      `${k}${rest}.</div>`;
  }

  // Was nicht im Signal steckt, holt kein Modell zurück. Diese Grenze
  // war bisher unsichtbar — bei schlechtem Ergebnis blieb offen, ob das
  // Werkzeug oder die Aufnahme schuld war.
  let qualitaet = "";
  const q = d.qualitaet || daten.qualitaet;
  if (q && q.stufe !== "gut") {
    qualitaet = `<div class="sa-warn w-audio">🎙️ Aufnahmequalität ${
      q.stufe === "schlecht" ? "schwach" : "grenzwertig"} ` +
      `(Störabstand ${q.snr_db} dB). ${q.rat.map(esc).join(" ")}</div>`;
  }

  // Das Backend liefert bereits nur noch die nennenswerten Passagen. Hier
  // wird zusätzlich gekappt: eine Warnung, die man zu Ende scrollen muss,
  // liest niemand.
  let hinweis = "";
  if (daten.overlaps.length) {
    const ZEIGE = 5;
    const stellen = daten.overlaps.slice(0, ZEIGE)
      .map((o) => `${zeit(o.start)}–${zeit(o.ende)}`).join(", ");
    const rest = daten.overlaps.length - ZEIGE;
    hinweis = `<div class="sa-warn w-overlap">⚠️ Hier haben mehrere gleichzeitig ` +
      `gesprochen — der Wortlaut ist dort unsicher: ${stellen}` +
      `${rest > 0 ? ` und ${rest} weitere` : ""}.</div>`;
  }

  let abschluss;
  if (gespeichert) {
    // Herunterladen ist das Ziel des Ablaufs und daher die Hauptaktion.
    abschluss =
      `<div class="md-aktionen">` +
      `<button id="btn-md-laden" class="primary">Markdown herunterladen</button>` +
      `<button id="btn-md-teilen" class="ghost breit">Teilen …</button>` +
      `</div>`;
  } else {
    const bekannte = d.sprecher_bekannt || [];
    const liste = bekannte.length
      ? `<datalist id="namen-liste">` +
        bekannte.map((n) => `<option value="${esc(n)}">`).join("") +
        `</datalist>`
      : "";
    let felder = liste;
    for (const lb of labels) {
      felder +=
        `<label class="row namen-zeile">` +
        `<span class="punkt" style="background:${farben[lb]}"></span>` +
        `<input class="namen-feld" data-label="${esc(lb)}" type="text" ` +
        (bekannte.length ? `list="namen-liste" ` : "") +
        // Wiedererkannte Stimme: Name steht schon drin, statt ihn jede
        // Woche neu zu tippen.
        ((d.vorschlag || {})[lb] ? `value="${esc(d.vorschlag[lb])}" ` : "") +
        `placeholder="${esc(namen[lb])}" autocomplete="off"></label>`;
    }

    abschluss =
      `<div class="sa-h">Sprecher benennen</div>` +
      `<div class="dim">Optional — leere Felder behalten den Standardnamen.</div>` +
      felder +
      `<button id="btn-speichern" class="primary">Als Markdown speichern</button>`;
  }

  // Sichern und Herunterladen stehen bewusst oben: nach einem langen
  // Meeting soll man dafür nicht durch das ganze Protokoll scrollen.
  $("#ergebnis").innerHTML =
    `<div class="card">${meta}<div class="sa-bar">${balken}</div>` +
    `<div class="sa-legs">${legende}</div>${qualitaet}${abschluss}` +
    `${zeitleiste}${gespraech}${hinweis}${gelernt}</div>`;

  if (gespeichert) {
    $("#btn-md-laden").addEventListener("click", () => mdHerunterladen());
    const teilen = $("#btn-md-teilen");
    if (navigator.share) {
      teilen.addEventListener("click", async () => {
        const datei = new File([letztesMarkdown.text], letztesMarkdown.name,
                               { type: "text/markdown" });
        try {
          if (navigator.canShare && navigator.canShare({ files: [datei] })) {
            await navigator.share({ files: [datei], title: letztesMarkdown.name });
          } else {
            await navigator.share({ title: letztesMarkdown.name,
                                    text: letztesMarkdown.text });
          }
        } catch (e) { /* Abbruch durch Nutzer */ }
      });
    } else {
      teilen.hidden = true;
    }
  } else {
    $("#btn-speichern").addEventListener("click", speichern);
  }

  // Ein angetipptes Wort berichtigen. Das ist der einzige Eingang, über
  // den Wissen von außen ins System kommt — bisher lernte das Glossar
  // ausschließlich aus der eigenen Ausgabe und konnte damit nur die
  // eigenen Fehler bestätigen.
  for (const el of document.querySelectorAll(".wackelig")) {
    el.addEventListener("click", () => berichtigen(el));
  }
}

async function berichtigen(el) {
  const falsch = el.dataset.wort;
  const richtig = (prompt(`Statt „${falsch}“ richtig:`, falsch) || "").trim();
  if (!richtig || richtig === falsch) return;
  el.textContent = richtig;
  el.classList.remove("wackelig");
  el.classList.add("berichtigt");
  try {
    const d = await rufe("/lernen", [schluessel, falsch, richtig]);
    toast(d.ok ? `Gemerkt: „${falsch}“ → „${richtig}“`
               : (d.fehler || "Konnte nicht gemerkt werden."));
  } catch (e) {
    toast("Konnte nicht gemerkt werden: " + e.message);
  }
}


async function speichern() {
  if (!letzteAnalyse) return;
  const eigene = {};
  document.querySelectorAll(".namen-feld").forEach((f) => {
    if (f.value.trim()) eigene[f.dataset.label] = f.value.trim();
  });
  $("#btn-speichern").disabled = true;
  try {
    const d = await rufe("/speichern", [schluessel, letzteAnalyse, eigene]);
    if (!d.ok) throw new Error(d.fehler);
    letztesMarkdown = { text: d.markdown, name: d.pfad.split("/").pop() };
    letzteAnalyse.namen = { ...letzteAnalyse.namen, ...eigene };
    ergebnisAnzeigen(letzteAnalyse, true);
    toast("Als Markdown gespeichert");
  } catch (e) {
    toast("Speichern: " + e.message);
    const btn = $("#btn-speichern");
    if (btn) btn.disabled = false;
  }
}

function mdHerunterladen() {
  if (!letztesMarkdown) return;
  const blob = new Blob([letztesMarkdown.text], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = letztesMarkdown.name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- Glossar ----------

let glossar = { begriffe: [], sprecher: [] };

async function glossarLaden() {
  try {
    const g = await rufe("/glossar", [schluessel]);
    if (!g.ok) throw new Error(g.fehler);
    glossar = { begriffe: g.begriffe || [], sprecher: g.sprecher || [],
                zaehler: g.zaehler || {} };
    $("#g-begriffe").value = glossar.begriffe
      .map((b) => (glossar.zaehler[b] ? `${b} (${glossar.zaehler[b]})` : b))
      .join("\n");
    $("#g-sprecher").value = glossar.sprecher.join("\n");
  } catch (e) {
    toast("Glossar: " + e.message);
  }
}

async function glossarSpeichern(begriffe, sprecher) {
  const g = await rufe("/glossar_speichern", [schluessel, begriffe, sprecher]);
  if (!g.ok) throw new Error(g.fehler);
  glossar = { begriffe: g.begriffe || [], sprecher: g.sprecher || [] };
  return glossar;
}

$("#glossar-box").addEventListener("toggle", () => {
  if ($("#glossar-box").open && schluessel) glossarLaden();
});

$("#btn-glossar-speichern").addEventListener("click", async () => {
  const btn = $("#btn-glossar-speichern");
  btn.disabled = true;
  try {
    await glossarSpeichern($("#g-begriffe").value, $("#g-sprecher").value);
    toast(`Glossar gespeichert (${glossar.begriffe.length} Begriffe).`);
  } catch (e) {
    toast("Glossar: " + e.message);
  } finally {
    btn.disabled = false;
  }
});

// ---------- Verlauf ----------

async function verlaufLaden() {
  try {
    const d = await rufe("/verlauf", [schluessel]);
    if (!d.ok) throw new Error(d.fehler);
    verlaufAnzeigen(d.eintraege);
  } catch (e) {
    toast("Verlauf: " + e.message);
  }
}

function verlaufAnzeigen(eintraege) {
  $("#eintrag-detail").hidden = true;
  const liste = $("#verlauf-liste");
  liste.hidden = false;
  if (!eintraege.length) {
    liste.innerHTML = `<div class="card dim">Noch keine Aufzeichnungen.</div>`;
    return;
  }
  liste.innerHTML = "";
  for (const e of eintraege) {
    const btn = document.createElement("button");
    btn.className = "v-card";
    btn.innerHTML =
      `<div class="v-titel">${esc(e.titel)}</div>` +
      `<div class="v-meta"><span>👥 ${e.sprecher ?? "?"} Sprecher</span>` +
      `<span>🕒 ${zeit(e.dauer_s)} min</span></div>`;
    btn.addEventListener("click", () => eintragOeffnen(e.pfad));
    liste.appendChild(btn);
  }
}

async function eintragOeffnen(pfad) {
  try {
    const d = await rufe("/eintrag", [schluessel, pfad]);
    if (!d.ok) throw new Error(d.fehler);
    letztesMarkdown = { text: d.markdown, name: pfad.split("/").pop() };
    const { marked } = await import(
      "https://cdn.jsdelivr.net/npm/marked@12/+esm");
    $("#eintrag-inhalt").innerHTML = marked.parse(d.body || "");
    $("#verlauf-liste").hidden = true;
    $("#eintrag-detail").hidden = false;
  } catch (e) {
    toast("Eintrag: " + e.message);
  }
}

$("#btn-zurueck").addEventListener("click", () => {
  $("#eintrag-detail").hidden = true;
  $("#verlauf-liste").hidden = false;
});

$("#btn-download").addEventListener("click", () => mdHerunterladen());

// ---------- Start ----------

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}

// Direktlink-Anmeldung: ?key=...&hf=... übernimmt Schlüssel und HF-Token
// und entfernt beides sofort wieder aus der Adresszeile.
const urlParams = new URLSearchParams(location.search);
if (urlParams.get("key")) {
  schluessel = urlParams.get("key").trim();
  localStorage.setItem("sa_key", schluessel);
}
if (urlParams.get("hf")) {
  hfToken = urlParams.get("hf").trim();
  localStorage.setItem("sa_hf", hfToken);
}
if (urlParams.get("key") || urlParams.get("hf")) {
  history.replaceState(null, "", location.pathname);
}

$("#transkript").addEventListener("input", () => {
  const n = $("#transkript").value.trim().split(/\s+/).filter(Boolean).length;
  $("#transkript-info").textContent = n
    ? `${n} Wörter — wird den Sprechern zugeordnet statt neu erkannt.`
    : "";
  if (n) $("#transkript-box").open = true;
});

const gemerkteSprache = localStorage.getItem("sa_sprache");
if (gemerkteSprache !== null) $("#sprache").value = gemerkteSprache;
$("#hf-input").value = hfToken;
$("#kontext").value = localStorage.getItem("sa_kontext") || "";

if (!schluessel) {
  $("#login").hidden = false;
} else {
  loginPruefen(schluessel).catch(() => {
    localStorage.removeItem("sa_key");
    schluessel = "";
    $("#login").hidden = false;
  });
}
