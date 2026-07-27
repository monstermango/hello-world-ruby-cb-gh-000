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
  verlaufAnzeigen(antwort.eintraege);
  tokenStatusPruefen();
  fortsetzenAnbieten();
}

// Fragt das Backend, ob die Anfragen mit HF-Token ankommen. Ohne Token läuft
// alles auf dem sehr kleinen anonymen GPU-Kontingent.
let tokenAktiv = null;

async function tokenStatusPruefen() {
  const feld = $("#token-status");
  try {
    const s = await rufe("/status", [schluessel]);
    tokenAktiv = !!s.token;
  } catch (e) {
    tokenAktiv = null;
  }
  feld.hidden = tokenAktiv === null;
  feld.textContent = tokenAktiv ? "Konto-Kontingent" : "ohne Token";
  feld.classList.toggle("warn", tokenAktiv === false);
  $("#login-status").textContent = tokenAktiv
    ? "HF-Token aktiv — GPU läuft über dein Konto."
    : "Kein HF-Token aktiv — nur kleines anonymes GPU-Kontingent.";
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
    toast(tokenAktiv ? "Token aktiv — GPU läuft über dein Konto."
                     : "Angemeldet, aber ohne gültiges HF-Token.");
  } catch (e) {
    $("#login-fehler").textContent = e.message;
  } finally {
    $("#btn-login").disabled = false;
  }
});

$("#key-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#btn-login").click();
});

$("#btn-key").addEventListener("click", () => {
  $("#key-input").value = schluessel;
  $("#hf-input").value = hfToken;
  $("#login").hidden = false;
});

// ---------- Navigation ----------

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) =>
      b.classList.toggle("active", b === btn));
    const ziel = btn.dataset.view;
    $("#view-analyse").hidden = ziel !== "analyse";
    $("#view-glossar").hidden = ziel !== "glossar";
    $("#view-verlauf").hidden = ziel !== "verlauf";
    if (ziel === "verlauf") verlaufLaden();
    if (ziel === "glossar") glossarLaden();
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
      $("#rec-status").textContent = "Zum Aufnehmen tippen";
      const endung = mime === "audio/mp4" ? "m4a" : "webm";
      setzeAudio(new File(recChunks, `aufnahme.${endung}`, { type: mime }));
    };
    // Sekündlich Daten abholen, damit lange Aufnahmen nicht am Stück im
    // Arbeitsspeicher gehalten werden.
    recorder.start(5000);
    await wakeLockAnfordern();
    $("#btn-record").classList.add("aktiv");
    const startZeit = Date.now();
    recTimer = setInterval(() => {
      const s = (Date.now() - startZeit) / 1000;
      $("#rec-status").textContent =
        `Aufnahme läuft … ${zeit(s)} — zum Stoppen tippen`;
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

// Zweiter Durchgang: ein Sprachmodell glättet Hörfehler und Grammatik.
async function textGlaetten(id, melde) {
  for (let i = 0; ; i++) {
    const gl = await rufeHartnaeckig("/glaetten", [schluessel, id, i]);
    if (!gl.ok) throw new Error(gl.fehler);
    melde(gl.abschnitte > 1
      ? `Überarbeite Text — Abschnitt ${Math.min(i + 1, gl.abschnitte)} von ${gl.abschnitte} …`
      : "Überarbeite Text …");
    laufSpeichern({ id, phase: "glaetten", i: i + 1 });
    if (!gl.weiter) break;
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

    // Jeder Schritt ist eine eigene Anfrage: eine einzelne lange Anfrage
    // würde das Zeitlimit der GPU-Zuteilung überschreiten.
    melde("Lade Audio hoch …");
    const transkript = $("#transkript").value.trim();
    const kontext = $("#kontext").value.trim();
    localStorage.setItem("sa_kontext", kontext);
    const vor = await rufeHartnaeckig(
      "/vorbereiten", [schluessel, audioDatei, transkript, kontext]);
    if (!vor.ok) throw new Error(vor.fehler);
    const eigenerText = vor.modus === "transkript";
    laufSpeichern({ id: vor.id, i: 0, abschnitte: null, eigenerText });

    melde("Erkenne Sprecher …");
    const spr = $("#sprache").value;
    localStorage.setItem("sa_sprache", spr);
    const dia = await rufeHartnaeckig("/diarisieren",
                                      [schluessel, vor.id, n, spr]);
    if (!dia.ok) throw new Error(dia.fehler);
    laufSpeichern({ id: vor.id, i: 0, abschnitte: dia.abschnitte, eigenerText });

    await abschnitteVerarbeiten(vor.id, dia.abschnitte, eigenerText, 0, melde);
    await textGlaetten(vor.id, melde);
    melde("Stelle Ergebnis zusammen …");
    return await rufeHartnaeckig("/abschliessen", [schluessel, vor.id]);
  });
});

function fortsetzenAnbieten() {
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
    if (lauf.phase !== "glaetten") {
      await abschnitteVerarbeiten(lauf.id, lauf.abschnitte, lauf.eigenerText,
                                  lauf.i, melde);
    }
    await textGlaetten(lauf.id, melde);
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

  let gespraech = `<div class="sa-h">Gespräch</div>`;
  for (const s of daten.segmente) {
    if (!s.text) continue;
    gespraech +=
      `<div class="sa-msg"><div class="sa-rail" style="background:${farben[s.label]}"></div>` +
      `<div class="sa-msg-body"><div class="sa-msg-head">${esc(namen[s.label])} · ` +
      `${zeit(s.start)}–${zeit(s.ende)}</div>` +
      `<div class="sa-bubble">${esc(s.text)}</div></div></div>`;
  }

  let hinweis = "";
  if (daten.overlaps.length) {
    const stellen = daten.overlaps
      .map((o) => `${zeit(o.start)}–${zeit(o.ende)}`).join(", ");
    hinweis = `<div class="sa-warn">⚠️ Gleichzeitiges Sprechen bei: ${stellen}</div>`;
  }

  let abschluss;
  if (gespeichert) {
    abschluss =
      `<div class="row md-aktionen">` +
      `<button id="btn-md-teilen" class="ghost grow">Teilen</button>` +
      `<button id="btn-md-laden" class="ghost grow">Herunterladen</button>` +
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
        `placeholder="${esc(namen[lb])}" autocomplete="off"></label>`;
    }

    abschluss =
      `<div class="sa-h">Sprecher benennen</div>` +
      `<div class="dim">Optional — leere Felder behalten den Standardnamen.</div>` +
      felder +
      `<button id="btn-speichern" class="primary">Als Markdown speichern</button>`;
  }

  $("#ergebnis").innerHTML =
    `<div class="card">${meta}<div class="sa-bar">${balken}</div>` +
    `<div class="sa-legs">${legende}</div>${zeitleiste}${gespraech}${hinweis}` +
    `${abschluss}</div>`;

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
