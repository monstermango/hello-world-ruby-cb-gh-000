# Cloud-Deployment (Docker + Caddy)

Komplettes Setup für den Dauerbetrieb auf einem eigenen Server (VPS):
Web-App + ML-Server + automatisches HTTPS mit Zugriffsschutz.
Damit ist das Tool jederzeit vom Handy erreichbar.

## Architektur

```
Handy/Browser ──HTTPS──▶ Caddy (Auto-TLS, Basic-Auth)
                           └──▶ web (Ruby, Port 8000)
                                  └── /ml/* ──▶ ml (FastAPI + Engines, Port 8001)
```

Nur Caddy ist von außen erreichbar; Web- und ML-Container sprechen intern.
Die pyannote-Modelle werden im Volume `hf-cache` gecacht.

## Einrichtung (einmalig, ~15 Minuten)

1. **Server mieten** – z. B. Hetzner CPX21 (3 vCPU/4 GB RAM, ~8 €/Monat).
   Ubuntu wählen, Docker installieren: `curl -fsSL https://get.docker.com | sh`
2. **DNS**: A-Eintrag deiner (Sub-)Domain auf die Server-IP zeigen lassen,
   z. B. `audio.deine-domain.de`.
3. **Projekt aufs Servern holen und konfigurieren**:
   ```sh
   git clone <repo-url> && cd <repo>/live-audio-tool/deploy
   cp .env.example .env
   nano .env       # DOMAIN, Basic-Auth, optional HF_TOKEN
   ```
   Passwort-Hash für die `.env` erzeugen:
   ```sh
   docker run --rm caddy:2 caddy hash-password --plaintext 'DEIN-PASSWORT'
   ```
   (jedes `$` im Hash in der `.env` als `$$` schreiben!)
4. **Starten**:
   ```sh
   docker compose up -d --build
   ```

Fertig: `https://audio.deine-domain.de` auf dem Handy öffnen, mit
Benutzer/Passwort anmelden, Aufnahme starten.

## Betrieb

- **Status prüfen**: `https://<domain>/ml/health` – zeigt Engines und geladenes Modell.
- **Update einspielen**: `git pull && docker compose up -d --build`
- **Logs**: `docker compose logs -f ml`
- **pyannote aktivieren**: `HF_TOKEN` in der `.env` setzen (Modell-Lizenz auf
  Hugging Face akzeptieren, siehe [../diarization-server/README.md](../diarization-server/README.md)),
  dann `docker compose up -d`. Der erste Analyse-Aufruf lädt das Modell (einmalig).

## Hinweise

- Der ML-Container nutzt CPU-PyTorch (schlankes Image). Auf einem Server ohne
  Zugriff auf `download.pytorch.org` stattdessen bauen mit:
  `docker compose build --build-arg TORCH_INDEX=https://pypi.org/simple`
  (größeres Image, gleiche Funktion).
- pyannote auf CPU analysiert grob in Echtzeit – für die Nachanalyse einer
  Besprechung völlig ausreichend. Eine GPU lohnt erst bei sehr viel Audio.
- Aufnahmen werden nur zur Analyse hochgeladen und nicht auf dem Server
  gespeichert (Verarbeitung in temporären Dateien).
