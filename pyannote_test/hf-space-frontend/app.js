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

async function rufe(endpunkt, daten) {
  if (!client) {
    const { Client } = await import(
      "https://cdn.jsdelivr.net/npm/@gradio/client/+esm");
    const optionen = hfToken ? { hf_token: hfToken } : {};
    client = await Client.connect(BACKEND, optionen);
  }
  const res = await client.predict(endpunkt, daten);
  return res.data[0];
}

// ---------- Login ----------

async function loginPruefen(neuerKey) {
  const antwort = await rufe("/verlauf", [neuerKey]);
  if (!antwort.ok) throw new Error(antwort.fehler || "Anmeldung fehlgeschlagen");
  schluessel = neuerKey;
  localStorage.setItem("sa_key", neuerKey);
  $("#login").hidden = true;
  verlaufAnzeigen(antwort.eintraege);
}

$("#btn-login").addEventListener("click", async () => {
  const wert = $("#key-input").value.trim();
  if (!wert) return;
  hfToken = $("#hf-input").value.trim();
  if (hfToken) localStorage.setItem("sa_hf", hfToken);
  client = null;
  $("#login-fehler").textContent = "";
  $("#btn-login").disabled = true;
  try {
    await loginPruefen(wert);
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
  $("#key-input").value = "";
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
    $("#view-verlauf").hidden = ziel !== "verlauf";
    if (ziel === "verlauf") verlaufLaden();
  });
});

// ---------- Aufnahme & Datei ----------

function setzeAudio(datei) {
  audioDatei = datei;
  $("#audio-name").textContent = datei.name;
  $("#audio-preview").src = URL.createObjectURL(datei);
  $("#audio-panel").hidden = false;
  $("#ergebnis").innerHTML = "";
}

$("#btn-datei").addEventListener("click", () => $("#file-input").click());
$("#file-input").addEventListener("change", (e) => {
  if (e.target.files[0]) setzeAudio(e.target.files[0]);
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
    recorder = new MediaRecorder(stream, { mimeType: mime });
    recChunks = [];
    recorder.ondataavailable = (e) => recChunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      clearInterval(recTimer);
      $("#btn-record").classList.remove("aktiv");
      $("#rec-status").textContent = "Zum Aufnehmen tippen";
      const endung = mime === "audio/mp4" ? "m4a" : "webm";
      setzeAudio(new File(recChunks, `aufnahme.${endung}`, { type: mime }));
    };
    recorder.start();
    $("#btn-record").classList.add("aktiv");
    const startZeit = Date.now();
    recTimer = setInterval(() => {
      $("#rec-status").textContent =
        `Aufnahme läuft … ${zeit((Date.now() - startZeit) / 1000)} — zum Stoppen tippen`;
    }, 500);
  } catch (e) {
    toast("Kein Mikrofonzugriff: " + e.message);
  }
});

// ---------- Analyse ----------

$("#btn-analyse").addEventListener("click", async () => {
  if (!audioDatei) return;
  $("#btn-analyse").disabled = true;
  $("#progress").hidden = false;
  $("#ergebnis").innerHTML = "";
  try {
    const n = parseInt($("#num-speakers").value, 10) || 0;
    const d = await rufe("/analysieren", [schluessel, audioDatei, n]);
    if (!d.ok) throw new Error(d.fehler);
    letzteAnalyse = d;
    ergebnisAnzeigen(d, false);
  } catch (e) {
    toast("Fehler: " + e.message);
  } finally {
    $("#btn-analyse").disabled = false;
    $("#progress").hidden = true;
  }
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
    let felder = "";
    for (const lb of labels) {
      felder +=
        `<label class="row namen-zeile">` +
        `<span class="punkt" style="background:${farben[lb]}"></span>` +
        `<input class="namen-feld" data-label="${esc(lb)}" type="text" ` +
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

if (!schluessel) {
  $("#login").hidden = false;
} else {
  loginPruefen(schluessel).catch(() => {
    localStorage.removeItem("sa_key");
    schluessel = "";
    $("#login").hidden = false;
  });
}
