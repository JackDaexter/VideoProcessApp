# ────────────────────────────────────────────────────────────────────────────
# PhantomPilot — VideoProcessApp Dockerfile
#
# Based on the real OpenShorts stack:
#   - faster-whisper (transcription)
#   - Google Gemini (clip selection + metadata)
#   - PySceneDetect (scene detection)
#   - MediaPipe + YOLOv8 (face tracking for 9:16 crop)
#   - FFmpeg (video cutting + encoding + caption burn-in)
#   - OpenCV (frame processing)
#   - FastAPI + Uvicorn (web server)
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    # FFmpeg build deps
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download YOLOv8n weights directly to a fixed absolute path — no cache resolution at runtime
RUN mkdir -p /app/models && \
    wget -q -O /app/models/yolov8n.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt

# Pre-download faster-whisper base model
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="PhantomPilot VideoProcessApp"
LABEL org.opencontainers.image.description="OpenShorts-based video processing: Clip Generator, AI Shorts, YouTube Studio"
LABEL org.opencontainers.image.version="2.0.0"

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env (with all Python packages)
COPY --from=builder /opt/venv /opt/venv
# Copy YOLOv8n weights from the explicit build path
COPY --from=builder /app/models /app/models
# Copy pre-downloaded faster-whisper model weights
COPY --from=builder /root/.cache /root/.cache
ENV PATH="/opt/venv/bin:$PATH"

# Absolute path to the pre-baked YOLO model — used by openshots_service.py
ENV YOLO_MODEL_PATH=/app/models/yolov8n.pt

# Non-root user — owns the models dir and whisper cache
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app/models /root/.cache

WORKDIR /app
COPY app/ ./app/

RUN mkdir -p /tmp/video_process && chown appuser:appuser /tmp/video_process

USER appuser

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
