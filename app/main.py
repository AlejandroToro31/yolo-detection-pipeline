"""
PPE Detection API — YOLOv8 FastAPI Microservice
================================================
Real-time Personal Protective Equipment detection endpoint.
Accepts image uploads via HTTP POST and returns structured
bounding box predictions as JSON.

Architecture:
    - Framework  : FastAPI (ASGI, async)
    - Model      : YOLOv8n via Ultralytics
    - Inference  : asyncio.to_thread (non-blocking CPU/GPU compute)
    - Validation : Pydantic v2 response schemas

Endpoints:
    GET  /          → API metadata
    GET  /health    → Liveness check
    GET  /ready     → Readiness check (model loaded)
    POST /api/v1/detect → PPE detection inference

Environment Variables:
    MODEL_PATH      : Path to best.pt artifact (default: models/best.pt)
    CONF_THRESHOLD  : Detection confidence cutoff (default: 0.40)
    IOU_THRESHOLD   : NMS IoU threshold (default: 0.50)

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

# ── Standard Library
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

# ── Third Party
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO


# ════════════════════════════════════════════════════════
# 1. LOGGING INFRASTRUCTURE
# ════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [YOLO-API] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YOLO-API")


# ════════════════════════════════════════════════════════
# 2. GLOBAL CONFIGURATION
# ════════════════════════════════════════════════════════

MODEL_PATH: str       = os.getenv("MODEL_PATH", "models/best.pt")

# ATTETION: As im uploading this file to HuggingFace, repo is going to be flat structure
#MODEL_PATH: str       = os.getenv("MODEL_PATH", "best.pt")

CONF_THRESHOLD: float = float(os.getenv("CONF_THRESHOLD", "0.40"))
IOU_THRESHOLD: float  = float(os.getenv("IOU_THRESHOLD", "0.50"))
MAX_PAYLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Singleton store — model loaded once at startup, shared across all requests
ml_state: Dict = {}


# ════════════════════════════════════════════════════════
# 3. SERVER LIFESPAN — MODEL SINGLETON PATTERN
# ════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager — controls model lifecycle.

    STARTUP:
        Loads YOLOv8 model into memory exactly once when the server boots.
        Runs warmup inference to compile CUDA kernels before first real request.
        Stores model in ml_state dict — shared across all request handlers.

    SHUTDOWN:
        Clears ml_state and releases memory cleanly.

    Why singleton pattern:
        Loading a 100MB+ model on every request would add seconds of latency.
        The singleton ensures one load at boot, then millisecond inference forever.
    """
    logger.info("Booting PPE Detection API...")
    logger.info(
        f"Config | Model: {MODEL_PATH} | "
        f"Conf: {CONF_THRESHOLD} | IoU: {IOU_THRESHOLD}"
    )

    # ── Model loading
    try:
        model = YOLO(MODEL_PATH)
        ml_state["model"] = model
        logger.info("YOLOv8 model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifact: {e}")
        raise RuntimeError(
            f"Server boot aborted — model artifact not found at: {MODEL_PATH}"
        ) from e

    # ── Warmup inference
    # Compiles CUDA kernels on first pass. Without warmup, the first real
    # inference call incurs a 3-5x latency spike from kernel compilation.
    logger.info("Running warmup inference...")
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    await asyncio.to_thread(model.predict, source=dummy_frame, verbose=False)
    logger.info("Model warmed up. API ready to serve requests.")

    yield  # ── Server is live and handling requests

    # ── Shutdown cleanup
    logger.info("Shutting down. Releasing model resources...")
    ml_state.clear()
    logger.info("Shutdown complete.")


# ════════════════════════════════════════════════════════
# 4. API INSTANTIATION
# ════════════════════════════════════════════════════════

app = FastAPI(
    title="PPE Detection API",
    description=(
        "Real-time YOLOv8 microservice for Personal Protective Equipment "
        "detection on factory and construction site camera feeds."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════
# 5. RESPONSE SCHEMAS — PYDANTIC CONTRACTS
# ════════════════════════════════════════════════════════

class Detection(BaseModel):
    """Single object detection result."""
    class_id   : int
    class_name : str
    confidence : float          # rounded to 4 decimal places
    bbox       : List[float]    # [xmin, ymin, xmax, ymax] pixel coordinates


class DetectionResponse(BaseModel):
    """Full inference response payload."""
    filename          : str
    total_detections  : int
    process_time_ms   : float
    detections        : List[Detection]


class HealthResponse(BaseModel):
    """Liveness check response."""
    status: str


class ReadyResponse(BaseModel):
    """Readiness check response."""
    status      : str
    model_path  : str
    conf        : float
    iou         : float


# ════════════════════════════════════════════════════════
# 6. UTILITY ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", tags=["Utility"])
async def root() -> dict:
    """API metadata — entry point for documentation discovery."""
    return {
        "api"     : "PPE Detection API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "health"  : "/health",
        "ready"   : "/ready",
        "detect"  : "/api/v1/detect",
    }


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health() -> HealthResponse:
    """
    Liveness check — confirms the API process is running.

    Used by Docker HEALTHCHECK and Kubernetes liveness probes.
    Returns 200 as long as the server process is alive.
    """
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse, tags=["Utility"])
async def ready() -> ReadyResponse:
    """
    Readiness check — confirms the model is loaded and inference is possible.

    Used by Kubernetes readiness probes to gate traffic routing.
    Returns 503 if the model is not yet loaded (e.g. still warming up).

    Liveness vs Readiness:
        /health  → Is the process alive? (restart if fails)
        /ready   → Is the model ready?  (stop routing traffic if fails)
    """
    if ml_state.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server may still be initializing."
        )
    return ReadyResponse(
        status     ="ready",
        model_path =MODEL_PATH,
        conf       =CONF_THRESHOLD,
        iou        =IOU_THRESHOLD,
    )


# ════════════════════════════════════════════════════════
# 7. INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════

@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Inference"])
async def detect_objects(file: UploadFile = File(...)) -> DetectionResponse:
    """
    PPE object detection endpoint.

    Accepts a multipart image upload, runs YOLOv8 inference, and returns
    all detected PPE objects with bounding box coordinates and confidence scores.

    Pipeline:
        1. MIME type validation       → reject non-image content types
        2. Payload size validation    → reject files > 10MB
        3. In-memory image decoding   → zero disk I/O via cv2.imdecode
        4. BGR → RGB conversion       → OpenCV reads BGR, YOLO expects RGB
        5. Thread-offloaded inference → asyncio.to_thread (non-blocking)
        6. Structured response        → Pydantic-validated JSON payload

    Args:
        file: Multipart image upload (JPEG, PNG, WebP)

    Returns:
        DetectionResponse with filename, latency, and list of Detection objects.

    Raises:
        400 : Invalid content type (non-image upload)
        413 : Payload exceeds 10MB limit
        422 : Valid image type but content cannot be decoded (corrupted file)
        500 : Unexpected inference error
        503 : Model not loaded
    """

    # ── Model availability check
    model = ml_state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Server may still be initializing."
        )

    # ── Step 1: MIME type validation
    # Note: content_type is client-controlled — actual content is validated
    # at decode time via cv2.imdecode (Step 4)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid content type: '{file.content_type}'. "
                "Only image/jpeg, image/png, image/webp accepted."
            )
        )

    # ── Step 2: Payload size validation
    # Read bytes first, then validate size — avoids sync seek/tell on async file
    image_bytes: bytes = await file.read()
    if len(image_bytes) > MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Payload too large: {len(image_bytes) / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {MAX_PAYLOAD_BYTES / 1024 / 1024:.0f}MB."
            )
        )

    try:
        # ── Step 3: In-memory image decoding (zero disk I/O)
        # np.frombuffer reads raw bytes as 1D array
        # cv2.imdecode decodes into H×W×3 numpy array
        nparr: np.ndarray = np.frombuffer(image_bytes, np.uint8)
        img: Optional[np.ndarray] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── Step 4: Content validation + BGR → RGB conversion
        # img is None if content_type was spoofed or file is corrupted
        if img is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Image decoding failed. File may be corrupted, "
                    "truncated, or content type was incorrectly declared."
                )
            )

        # OpenCV reads BGR — YOLO expects RGB
        # Missing this conversion produces silently wrong predictions
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Step 5: Thread-offloaded inference
        # asyncio.to_thread offloads CPU/GPU-bound inference to a thread pool,
        # keeping the event loop free to accept other requests during computation.
        # Never run model.predict() directly in async def — it blocks the loop.
        start_time: float = time.perf_counter()

        results = await asyncio.to_thread(
            model.predict,
            source      =img,
            conf        =CONF_THRESHOLD,
            iou         =IOU_THRESHOLD,
            verbose     =False,
        )
        results = results[0]

        latency_ms: float = round((time.perf_counter() - start_time) * 1000, 2)

        # ── Step 6: Parse detections into Pydantic-validated payload
        detections: List[Detection] = [
            Detection(
                class_id   =int(box.cls[0]),
                class_name =model.names[int(box.cls[0])],
                confidence =round(float(box.conf[0]), 4),
                bbox       =[round(float(x), 2) for x in box.xyxy[0]],
            )
            for box in results.boxes
        ]

        logger.info(
            f"Inference complete | File: '{file.filename or 'unknown'}' | "
            f"Detections: {len(detections)} | Latency: {latency_ms}ms"
        )

        return DetectionResponse(
            filename         =file.filename or "unknown",
            total_detections =len(detections),
            process_time_ms  =latency_ms,
            detections       =detections,
        )

    except HTTPException:
        # Re-raise HTTPExceptions — they are intentional, not unexpected errors
        raise

    except Exception as e:
        logger.error(f"Unexpected inference error on '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during inference. Check server logs."
        )

    finally:
        # Explicit memory release — prevents tensor accumulation in long-running
        # servers processing high-volume camera feeds
        try:
            del image_bytes, nparr, img
        except NameError:
            pass  # Variables may not exist if error occurred before assignment