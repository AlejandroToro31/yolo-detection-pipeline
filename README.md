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

A production-deployed object detection microservice for automated workplace safety monitoring. Detects the presence and absence of Personal Protective Equipment (PPE) — specifically hard hats and high-visibility vests — from images of construction sites.

Built around the kinds of PPE covered by EU workplace-safety rules (e.g. Regulation (EU) 2016/425). This is a portfolio project, not a certified safety system.

🔗 **Live Demo:** [https://hunter1a11-realtime-safety-detection.hf.space/docs](https://hunter1a11-realtime-safety-detection.hf.space/docs) — try inference directly in your browser via the interactive Swagger UI.

> **Note on "real-time":** the underlying YOLOv8n model is architecturally suited for real-time inference (single-stage, ~100+ FPS on GPU benchmarks). The current deployment is a single-image HTTP API — see the [Roadmap](#roadmap) for what a continuous video-stream deployment would require.

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
├── research/
│   └── ppe_vision_system.py     # YOLOv8 training pipeline (Roboflow + Ultralytics)
├── deploy/
│   └── huggingface/
│       └── Dockerfile           # Hugging Face Space variant (flat layout, port 7860)
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

Captured live from the deployed API on a real construction-site photo with three workers:

```json
{
  "filename": "kO4HhmY4nP5ulWvr8Ypu2qAPNA.jpg",
  "total_detections": 6,
  "process_time_ms": 104.17,
  "detections": [
    { "class_id": 9, "class_name": "vest",   "confidence": 0.8395, "bbox": [1036.45, 535.96, 1171.97, 718.72] },
    { "class_id": 9, "class_name": "vest",   "confidence": 0.83,   "bbox": [796.13, 509.14, 930.01, 709.14] },
    { "class_id": 9, "class_name": "vest",   "confidence": 0.8226, "bbox": [627.76, 570.64, 724.72, 773.58] },
    { "class_id": 3, "class_name": "helmet", "confidence": 0.763,  "bbox": [1035.41, 450.19, 1111.06, 505.07] },
    { "class_id": 3, "class_name": "helmet", "confidence": 0.7582, "bbox": [818.96, 424.48, 901.46, 488.17] },
    { "class_id": 3, "class_name": "helmet", "confidence": 0.6952, "bbox": [664.63, 489.98, 735.93, 552.34] }
  ]
}
```

**What this result shows:** all three workers in the photo wear a helmet and a high-visibility vest, and the API returns exactly that — three helmets, three vests, no violation flagged. This is one test image, not a benchmark.

**Latency:** 104.17ms for this request on CPU (Hugging Face Spaces free tier, shared vCPU). Latency varies between runs on the shared tier — measurements so far have ranged from roughly 104ms to 138ms — so treat it as an approximate figure.

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

**OS dependencies:** `python:3.10-slim` strips many system libraries for size. `libgl1` and `libglib2.0-0` are reinstalled because `ultralytics` also installs the full `opencv-python` package alongside `opencv-python-headless`, and the full build needs libGL. Removing the duplicate OpenCV install is a known cleanup item.

**Non-root execution:** The container runs as `api_user` — a non-root system account. Principle of least privilege.

**Health monitoring:** Docker's native `HEALTHCHECK` polls `/health` every 30 seconds with a 60-second startup grace period for model loading.

---

## Repository vs Hugging Face Space

The same API runs in two places. The code is the same; the packaging differs, because the Space is a separate Git repository on Hugging Face and its web uploader cannot create folders.

| | GitHub repository | Hugging Face Space |
|---|---|---|
| Layout | `app/main.py`, `models/`, `training/` | flat: `main.py`, `best.pt`, `Dockerfile`, `requirements.txt`, `README.md` at the root |
| Model weights | not committed (`*.pt` is git-ignored) — download link in Quick Start | `best.pt` (under 10 MB) committed to the Space |
| Dockerfile | `Dockerfile` — copies `app/` and `models/`, runs `app.main:app` | `deploy/huggingface/Dockerfile` — copies `main.py` and `best.pt`, runs `main:app` |
| Port | 8000 | 7860 (Spaces default) |
| Workers | 4 | 1 (shared free-tier CPU) |
| `MODEL_PATH` | `models/best.pt` | `best.pt`, set by `ENV` in the Dockerfile |
| README | plain Markdown | starts with the YAML block Spaces reads (`sdk: docker`); GitHub shows it as a small table |

`main.py` and `requirements.txt` are identical in both. The one environment-specific setting, `MODEL_PATH`, is set by the Dockerfile `ENV` line rather than by editing code. Precedence: `docker run -e` > Dockerfile `ENV` > the default in `os.getenv`.

**Why Hugging Face Spaces:** the first deployment attempt, on Render's free tier (512 MB RAM), failed with an out-of-memory error — PyTorch plus Ultralytics need more than that. The Spaces free CPU tier has 16 GB RAM and ran the same container without changes beyond the port, layout and worker count above.

---

## Roadmap

**Current state:** single-image HTTP inference API. The trained model and inference logic transfer completely unchanged to a video pipeline — this is deliberately the hard part, and it's already done.

**What a continuous real-time video deployment would add:**
- **Frame ingestion loop** — `cv2.VideoCapture` or an RTSP stream, replacing request/response with continuous polling
- **Object tracking** (ByteTrack) — maintains identity across frames, robust to brief occlusion
- **ONNX → TensorRT → INT8 export** — strips Python runtime overhead, hardware-tuned kernels, ~4× smaller model with minimal accuracy loss — the standard path for edge deployment on hardware like NVIDIA Jetson

---

## Changelog

**v1.0.1 — colour-channel fix.** The API used to convert decoded images from BGR to RGB before calling `model.predict()`. Ultralytics treats numpy input as BGR, so the model was receiving swapped channels. It showed up as a visibly wrong result: on the test photo above, a worker wearing a vest was flagged `no-vest`. After removing the conversion, the same photo gives 6 correct detections: the false `no-vest` is gone, the left-hand worker's helmet and vest (previously missed) are detected, and confidences rose on the others (e.g. the right-hand worker's vest from 0.71 to 0.84). Single-image comparison, not a benchmark.