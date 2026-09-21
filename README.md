---
title: PPE Safety Detection API
emoji: 🦺
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
---

# YOLOv8 PPE Detection API

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A production-deployed object detection microservice for automated workplace safety monitoring. Detects the presence and absence of Personal Protective Equipment (PPE) — specifically hard hats and high-visibility vests — from construction site camera feeds.

Compliant with EU workplace safety standards (SUVA / EU PPE Directive 2016/425).

🔗 **Live Demo:** [https://hunter1a11-realtime-safety-detection.hf.space/docs](https://hunter1a11-realtime-safety-detection.hf.space/docs) — try inference directly in your browser via the interactive Swagger UI.

> **Note on "real-time":** the underlying YOLOv8n model is architecturally suited for real-time inference (single-stage, ~100+ FPS on GPU benchmarks). The current deployment is a single-image HTTP API — see [Roadmap](#roadmap) for what a continuous video-stream deployment would require.

---

## System Architecture

| Component | Implementation | Details |
|-----------|---------------|---------|
| **Detection Engine** | YOLOv8 Nano | Custom-trained, anchor-free single-stage detector |
| **Web Framework** | FastAPI + Uvicorn | ASGI, async request handling |
| **Inference** | asyncio.to_thread | Non-blocking CPU inference — event loop stays free |
| **Image Decoding** | cv2.imdecode | Zero disk I/O — raw bytes decoded directly in RAM |
| **Payload Design** | Coordinates only | Backend returns raw bbox coordinates — frontend renders boxes |
| **Container** | python:3.10-slim | Non-root user, layer-cached builds, HEALTHCHECK |
| **Inference Device** | CPU | Optimized for cost-efficient cloud deployment |

**Why CPU deployment:**
GPU inference saves ~20ms per request but network latency alone is 50-100ms. For static image requests via HTTP, CPU is the correct engineering choice — 10× cheaper, instant cold starts, deployable anywhere without NVIDIA driver dependencies. GPU becomes worthwhile specifically for continuous real-time video stream processing at 30+ FPS, where compute is repeated back-to-back with no network round-trip between frames — see Roadmap below.

---

## Tech Stack

- **Deep Learning:** PyTorch 2.1, Ultralytics YOLOv8
- **Web Server:** FastAPI 0.104, Uvicorn (with uvloop + httptools)
- **Computer Vision:** OpenCV (`opencv-python-headless`), NumPy, Pillow
- **DevOps:** Docker, python:3.10-slim base image

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API metadata and endpoint discovery |
| `GET` | `/health` | Liveness check — is the process running? |
| `GET` | `/ready` | Readiness check — is the model loaded? |
| `POST` | `/api/v1/detect` | PPE object detection inference |

---

## Project Structure

```
ppe-detection-api/
├── app/
│   └── main.py                  # FastAPI inference endpoint
├── models/
│   └── best.pt                  # Model artifact (download separately — see below)
├── training/
│   └── ppe_vision_system.py     # YOLOv8 training pipeline (Roboflow + Ultralytics)
├── assets/
│   └── ppe_result.mp4           # Demo video — Swagger UI inference walkthrough
├── Dockerfile                   # Production container definition
├── requirements.txt             # Pinned Python dependencies
└── README.md
```

---

## Quick Start

### 1. Download the Model Artifact

The trained model weights are stored externally to keep the repository lightweight.

1. Download `best.pt` from: [Model Registry (Google Drive)](https://drive.google.com/file/d/1RvJ6Xt1OUKbKwuQTsZ6ynqUdbGNnIXQd/view?usp=sharing)
2. Place it inside the `models/` directory:

```
models/
└── best.pt
```

### 2. Build the Container

```bash
docker build -t ppe-detection-api:v1 .
```

First build takes ~5 minutes (dependency installation). Subsequent builds after code-only changes take ~2 seconds due to Docker layer caching.

### 3. Run the Container

Default configuration:

```bash
docker run -p 8000:8000 ppe-detection-api:v1
```

With runtime threshold overrides (adjust per lighting conditions):

```bash
docker run -p 8000:8000 \
  -e CONF_THRESHOLD=0.35 \
  -e IOU_THRESHOLD=0.50 \
  ppe-detection-api:v1
```

### 4. Run Inference

Navigate to **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

Upload an image to `POST /api/v1/detect` and inspect the JSON response.

Verify the API is healthy:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

---

## Example Response — Real Deployment Output

Captured live from the deployed API, testing a real multi-person construction site photo:

```json
{
  "filename": "construction_site_workers.jpg",
  "total_detections": 4,
  "process_time_ms": 137.64,
  "detections": [
    {
      "class_id": 3,
      "class_name": "helmet",
      "confidence": 0.7113,
      "bbox": [1034.04, 449.12, 1113.21, 508.91]
    },
    {
      "class_id": 9,
      "class_name": "vest",
      "confidence": 0.7073,
      "bbox": [1045.73, 533.35, 1168.94, 719.48]
    },
    {
      "class_id": 3,
      "class_name": "helmet",
      "confidence": 0.6935,
      "bbox": [817.54, 423.32, 904.45, 493.49]
    },
    {
      "class_id": 8,
      "class_name": "no-vest",
      "confidence": 0.5201,
      "bbox": [801.55, 512.16, 932.71, 711.87]
    }
  ]
}
```

**What this result shows:** two workers correctly detected wearing helmets; one worker correctly identified as wearing a compliant safety vest; a second worker in the same frame correctly flagged as **not** wearing one. The model distinguishes compliance on a per-person basis within a single crowded scene, not just a single aggregate judgment.

**Real-world performance:** 137.64ms end-to-end latency on CPU (Hugging Face Spaces free tier, shared vCPU).

`bbox` format: `[xmin, ymin, xmax, ymax]` in pixel coordinates.

The backend returns raw spatial coordinates only — bounding box rendering is handled client-side. This keeps inference latency minimal and decouples visualization from detection logic.

---

## Training Pipeline

To retrain on a custom dataset:

1. Upload your labeled dataset to [Roboflow](https://roboflow.com) in YOLOv8 format
2. Set your Roboflow API key as an environment variable:

```bash
export ROBOFLOW_API_KEY=your_key_here
```

3. Configure and run the training pipeline:

```python
from training.ppe_vision_system import PPEVisionSystem, YOLOConfig

config = YOLOConfig(
    ROBOFLOW_WORKSPACE="your-workspace",
    ROBOFLOW_PROJECT="your-project",
    ROBOFLOW_VERSION=1,
    EPOCHS=100,
    BATCH_SIZE=16,
)

system = PPEVisionSystem(config)
system.train()
```

The pipeline automatically saves the best checkpoint by validation mAP. The resulting `best.pt` artifact can be placed directly in `models/` for deployment.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.pt` | Path to trained model artifact |
| `CONF_THRESHOLD` | `0.40` | Minimum confidence for detection |
| `IOU_THRESHOLD` | `0.50` | NMS IoU overlap threshold |

All variables can be overridden at runtime via `docker run -e VAR=value` without modifying or rebuilding the container.

---

## Docker Notes

**OS dependencies:** `python:3.10-slim` strips many system libraries for size. `libgl1` and `libglib2.0-0` are reinstalled — userspace libraries required by OpenCV's image processing backend, even with the headless build.

**Non-root execution:** The container runs as `api_user` — a non-root system account. Principle of least privilege.

**Health monitoring:** Docker's native `HEALTHCHECK` polls `/health` every 30 seconds with a 60-second startup grace period for model loading.

---

## Live Deployment Notes

This API is deployed live on [Hugging Face Spaces](https://huggingface.co/spaces) (Docker SDK, free tier — 16GB RAM, 2 vCPUs). Unlike typical PaaS free tiers (Render, Railway) which cap RAM at 512MB — insufficient for PyTorch + Ultralytics — Hugging Face Spaces is purpose-built for ML workloads and handles this stack natively.

Two deployment-specific adjustments from the standard Docker setup above:
- **Port:** Hugging Face Spaces expects the container to listen on port `7860`, not `8000`
- **Single worker:** `--workers 1` in the CMD instruction, since the free tier is a shared environment

---

## Roadmap

**Current state:** single-image HTTP inference API. The trained model and inference logic transfer completely unchanged to a video pipeline — this is deliberately the hard part, and it's already done.

**What a continuous real-time video deployment would add:**
- **Frame ingestion loop** — `cv2.VideoCapture` or an RTSP stream, replacing request/response with continuous polling
- **Object tracking** (ByteTrack) — maintains identity across frames, robust to brief occlusion
- **ONNX → TensorRT → INT8 export** — strips Python runtime overhead, hardware-tuned kernels, ~4× smaller model with minimal accuracy loss — the standard path for edge deployment on hardware like NVIDIA Jetson