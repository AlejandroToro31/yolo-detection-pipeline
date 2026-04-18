# YOLOv8 Real-Time PPE Detection API: Autonomous Safety Vision

An enterprise-grade, ultra-low latency Object Detection microservice engineered for automated OSHA safety compliance. 

This API ingests live video frame streams from construction site cameras and utilizes a custom-trained YOLOv8 Nano architecture to instantly detect the presence—and critical absence—of Personal Protective Equipment (PPE), specifically hard hats and high-visibility vests.

## System Architecture & Telemetry

* **The Engine:** Ultralytics YOLOv8 Nano, fine-tuned on 5,000 highly occluded industrial images.
* **The Speed:** Achieved a peak inference latency of **2.2ms (454 FPS)**, enabling true zero-latency edge deployment.
* **The Payload Decoupling:** Strict separation of concerns. To conserve network bandwidth, the backend never renders bounding boxes onto the matrices. It executes direct-memory image decoding and returns a pure JSON array of spatial coordinates and confidence scores for frontend consumption.
* **The Infrastructure:** A hardened `python:3.10-slim` Docker container. The build process features a custom OS-level patch injecting `libgl1` and `libglib2.0-0` directly into the Debian kernel to support headless C++ OpenCV, preventing the massive bloat of standard GUI libraries.
* **DevSecOps:** The R&D training pipeline strictly utilized cloud secrets managers to inject API credentials dynamically, ensuring zero hardcoded vulnerabilities in the repository.

## Tech Stack
* **Deep Learning:** PyTorch, Ultralytics (YOLOv8)
* **Web Server:** FastAPI, Uvicorn
* **Data Processing:** OpenCV (`opencv-python-headless`), NumPy
* **DevOps:** Docker

## Quick Start (Production Environment)

### Download the Model Artifact
To maintain repository efficiency, the custom-trained YOLO weights are hosted in external artifact storage. 

1. Download `best.pt` from this direct link: [External Model Registry (Google Drive)](https://drive.google.com/file/d/1RvJ6Xt1OUKbKwuQTsZ6ynqUdbGNnIXQd/view?usp=sharing)

2. Place the artifact strictly inside the local `models/` directory.

### Compile the Infrastructure
```bash
docker build -t yolo-api:latest .
```
### 2. Initialize the Inference Node
The API utilizes Non-Maximum Suppression (NMS). You can dynamically adjust the algorithmic thresholds at runtime via environment variables depending on the camera's lighting conditions:

```bash
docker run -p 8000:8000 -e CONF_THRESHOLD=0.40 -e IOU_THRESHOLD=0.50 yolo-api:latest
```

### 3. Execute Inference & Telemetry
Navigate to http://127.0.0.1:8000/docs to access the interactive Swagger UI. Upload a raw image to the POST /detect/ endpoint.

## Expected JSON Telemetry Response:

```JSON
{
  "filename": "worker_cam_04.jpg",
  "total_detections": 2,
  "detections": [
    {
      "class_id": 3,
      "class_name": "helmet",
      "confidence": 0.838,
      "bbox": [103.06, 228.40, 359.05, 392.52]
    },
    {
      "class_id": 8,
      "class_name": "no-vest",
      "confidence": 0.761,
      "bbox": [357.15, 524.19, 500.53, 801.02]
    }
  ]
}
```