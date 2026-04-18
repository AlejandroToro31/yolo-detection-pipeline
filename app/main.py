import os
import time
import logging
import asyncio
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from ultralytics import YOLO

# --- LOGGING INFRASTRUCTURE ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [YOLO-API] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YOLO-API")

# --- GLOBAL STATE & CONFIGURATION ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.40"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.50"))
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10 MB limit
ml_state = {} 

# --- SERVER LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting YOLOv8 Real-Time Detection API...")
    
    try:
        model = YOLO(MODEL_PATH)
        ml_state["model"] = model
        logger.info(f"YOLOv8 Engine online. Confidence Threshold: {CONF_THRESHOLD} | IoU: {IOU_THRESHOLD}")
    except Exception as e:
        logger.critical(f"Failed to load model artifact: {e}")
        raise RuntimeError("Server boot aborted due to missing artifact.")
        
    yield 
    
    logger.info("Shutting down. Releasing resources...")
    ml_state.clear()

# --- API INSTANTIATION ---
app = FastAPI(
    title="PPE Detection API",
    description="Real-Time YOLOv8 microservice for Personal Protective Equipment tracking.",
    version="1.0.0",
    lifespan=lifespan
)

# --- RESPONSE SCHEMAS ---
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [xmin, ymin, xmax, ymax]

class DetectionResponse(BaseModel):
    filename: str
    total_detections: int
    process_time_ms: float
    detections: List[Detection]

# --- THE ENDPOINT ---
@app.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Image required.")

    # 1. Payload Size Validation
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_PAYLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Payload Exceeds 10MB limit.")

    try:
        # 2. Direct Memory Decode
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("OpenCV failed to decode image.")

        model = ml_state["model"]
        
        # 3. Asynchronous Inference (Thread Offloading & Telemetry)
        start_time = time.perf_counter()
        
        results = await asyncio.to_thread(
            model.predict,
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        results = results[0]
        
        end_time = time.perf_counter()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        # 4. Payload Extraction
        detections = []
        for box in results.boxes:
            detections.append(
                Detection(
                    class_id=int(box.cls[0]),
                    class_name=model.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=[float(x) for x in box.xyxy[0]]
                )
            )

        logger.info(f"Analyzed '{file.filename}': {len(detections)} objects in {latency_ms}ms.")

        # Explicitly release large array from memory
        del image_bytes, nparr, img

        return DetectionResponse(
            filename=file.filename,
            total_detections=len(detections),
            process_time_ms=latency_ms,
            detections=detections
        )

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during detection.")