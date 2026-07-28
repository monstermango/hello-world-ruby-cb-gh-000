self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(clients.claim()));

// Der Service Worker ist der einzige Teil der App, der läuft, wenn sie
// geschlossen ist. Nur hier kann eine Benachrichtigung ankommen.
self.addEventListener("push", (e) => {
  let d = { titel: "Sprecher-Analyse", text: "" };
  try {
    if (e.data) d = { ...d, ...e.data.json() };
  } catch (_) {
    // Nutzlast unlesbar — lieber eine karge Meldung als gar keine.
    if (e.data) d.text = e.data.text();
  }
  e.waitUntil(self.registration.showNotification(d.titel, {
    body: d.text,
    icon: "icon-192.png",
    badge: "icon-192.png",
    tag: "sa-analyse",       // ersetzt die vorige statt zu stapeln
    data: { url: "./" },
  }));
});

// Antippen bringt die App nach vorn, statt eine zweite Instanz zu öffnen.
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  e.waitUntil((async () => {
    const fenster = await clients.matchAll({ type: "window",
                                             includeUncontrolled: true });
    for (const f of fenster) {
      if ("focus" in f) return f.focus();
    }
    if (clients.openWindow) return clients.openWindow("./");
  })());
});
