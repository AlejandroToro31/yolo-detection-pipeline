# YOLOv8 Real-Time PPE Detection API: Autonomous Safety Vision

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Ultralytics](https://img.shields.io/badge/YOLOv8-1A1A1A?style=for-the-badge&logo=ultralytics)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)

An enterprise-grade, ultra-low latency Object Detection microservice engineered for automated OSHA safety compliance. 

This API ingests live video frame streams from construction site cameras and utilizes a custom-trained YOLOv8 Nano architecture to instantly detect the presence—and critical absence—of Personal Protective Equipment (PPE), specifically hard hats and high-visibility vests. 

## System Architecture & Telemetry

* **The Engine:** `Ultralytics` YOLOv8 Nano, trained on 5,000 highly occluded industrial images.
* **The Speed:** Achieved a peak inference latency of **2.2ms (454 Frames Per Second)**, enabling true zero-latency edge deployment.
* **The Payload:** Strict decoupling. To conserve bandwidth, the backend never draws the bounding boxes. It executes direct-memory image decoding and returns a pure mathematical JSON array of coordinates and confidence scores for frontend rendering.
* **The Infrastructure:** A hardened `python:3.10-slim` Docker container. The build process features a custom OS-level patch injecting `libgl1` and `libglib2.0-0` directly into the Debian kernel to support headless C++ OpenCV, preventing the massive bloat of standard GUI libraries.
* **DevSecOps:** The R&D training pipeline strictly utilized cloud secrets managers to inject API credentials dynamically, ensuring zero hardcoded vulnerabilities in the repository.

## Quick Start (Production Environment)

### 0. Download the Model Artifact
The custom-trained YOLOv8 weights are hosted externally to maintain repository efficiency.
1. Download `best.pt` from this direct link: **[https://drive.google.com/file/d/1RvJ6Xt1OUKbKwuQTsZ6ynqUdbGNnIXQd/view?usp=sharing]**
2. Place the artifact strictly inside the `models/` directory.

### 1. Compile the Infrastructure
```bash
docker build -t yolo-api:latest .
```

### 2. Ignite the Container
The API utilizes Non-Maximum Suppression (NMS). You can dynamically adjust the algorithmic thresholds at runtime depending on the camera's environmental conditions:

```Bash
docker run -p 8000:8000 -e CONF_THRESHOLD=0.40 -e IOU_THRESHOLD=0.50 yolo-api:latest
```
### 3. Execute Inference
Navigate to http://127.0.0.1:8000/docs to access the Swagger UI. Upload a raw image to the /detect/ endpoint.

# Expected JSON Telemetry Response:

JSON
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