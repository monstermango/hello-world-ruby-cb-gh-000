# Diarisierungs-Server: eingebaute Engine + pyannote
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY diarization-server/requirements.txt diarization-server/requirements-pyannote.txt ./

# CPU-Torch aus dem PyTorch-Index (spart >2 GB CUDA-Ballast).
# Über TORCH_INDEX übersteuerbar, falls der Index nicht erreichbar ist.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchaudio --index-url "$TORCH_INDEX" \
 && pip install --no-cache-dir -r requirements.txt -r requirements-pyannote.txt

COPY diarization-server/server.py ./

# pyannote-Modelle landen im Volume und überleben Container-Neustarts
ENV HF_HOME=/cache/huggingface
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
