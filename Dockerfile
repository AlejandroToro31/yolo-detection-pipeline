# ==============================================================
# PPE Detection API — Production Dockerfile
# ==============================================================
# YOLOv8 FastAPI microservice for real-time PPE detection.
# Optimized for CPU inference on lightweight Debian base.
#
# Build:
#   docker build -t ppe-detection-api:v1 .
#
# Run:
#   docker run -p 8000:8000 ppe-detection-api:v1
#
# Run with overrides:
#   docker run -p 8000:8000 \
#     -e CONF_THRESHOLD=0.35 \
#     -e MODEL_PATH=models/best.pt \
#     ppe-detection-api:v1
# ==============================================================

# ── Base Image
# GCR mirror: faster than Docker Hub, avoids rate limits
# python:3.10-slim: minimal Debian — significantly smaller than full Python image
FROM mirror.gcr.io/library/python:3.10-slim

# ── Image Metadata
LABEL maintainer="portfolio-project"
LABEL version="1.0.0"
LABEL description="YOLOv8 PPE Detection API — Real-time safety equipment detection"

# ── Python Runtime Configuration
# PYTHONDONTWRITEBYTECODE=1 : Prevents .pyc bytecode files — keeps container clean
# PYTHONUNBUFFERED=1        : Forces stdout/stderr to flush immediately — real-time Docker logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Application Configuration
# These are system defaults — override at runtime via docker run -e
# ENV hierarchy: docker run -e (highest) > Dockerfile ENV > os.getenv fallback (lowest)
#ENV MODEL_PATH=models/best.pt

# ATTENTION: As im uploading this file to HuggingFace, repo is going to be flat structure
ENV MODEL_PATH=best.pt
ENV CONF_THRESHOLD=0.40
ENV IOU_THRESHOLD=0.50

# ── OS-Level Dependencies
# OpenCV requires libgl1 and libglib2.0-0 on python:3.10-slim.
# These libraries are stripped from the slim base image but required
# by OpenCV's image processing backend — even with opencv-python-headless.
# curl is required for the Docker HEALTHCHECK instruction.
# --no-install-recommends: skips suggested packages, minimizes image footprint.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Security: Non-Root User
# Running as root inside a container is a security vulnerability.
# If the container is compromised, root access exposes the host system.
# -r : system account (no login shell, no cron)
# -m : creates home directory (~/.cache) required by PyTorch/YOLO weight caching
# Explicit groupadd ensures deterministic GID — avoids unpredictable system assignments
RUN groupadd -r api_user && useradd -m -r -g api_user api_user

# ── Working Directory
WORKDIR /workspace

# Set workspace ownership — api_user needs write access for YOLO cache and logs
RUN chown api_user:api_user /workspace

# ── Layer Caching Strategy
# Install dependencies BEFORE copying application code.
# If only main.py changes, Docker reuses the cached pip install layer.
# Full pip install only re-runs when requirements.txt changes.
# Build time: ~5 minutes (cold) vs ~2 seconds (cached code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code
# --chown sets file ownership in a single COPY layer.
# Avoids a separate RUN chown command which would add an extra layer.
#COPY --chown=api_user app/ app/
#COPY --chown=api_user models/ models/

# ATTENTION: As im uploading this file to HuggingFace, repo is going to be flat structure
COPY --chown=api_user main.py .
COPY --chown=api_user best.pt .

# ── Drop Privileges
# Switch to non-root user for all subsequent operations including CMD.
# All runtime processes run as api_user — principle of least privilege.
USER api_user

# ── Port Declaration
# Documents that the container listens on port 8000.
# Does not actually publish the port — use -p 8000:8000 at docker run.
EXPOSE 7860

# ── Health Monitoring
# Docker automatically monitors container health using this instruction.
# --interval=30s     : check every 30 seconds
# --timeout=10s      : fail if no response within 10 seconds
# --start-period=60s : grace period before checks start (model loading time)
# --retries=3        : mark unhealthy after 3 consecutive failures
# Connects directly to the /health liveness endpoint in main.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint
# --host 0.0.0.0 : bind to all interfaces — required for Docker port mapping
#                  (127.0.0.1 would be unreachable from outside the container)
# --workers 4    : 4 independent processes for parallel request handling
#                  Rule of thumb: (2 × CPU_cores) + 1
#                  Note: each worker loads its own model instance into memory
#                  Calculate: model_size_MB × workers ≤ available RAM
# --workers 1    : Choose 1 for HuggingFace Space upload
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ATTENTION: As im uploading this file to HuggingFace, repo is going to be flat structure
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]