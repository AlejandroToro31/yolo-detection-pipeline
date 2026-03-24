import os
import logging
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
ml_state = {} 

# --- SERVER LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting YOLOv8 Real-Time Detection API...")
    
    try:
        # Load the compiled Ultralytics artifact
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
    detections: List[Detection]

# --- THE ENDPOINT ---
@app.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid payload. Image required.")

    try:
        # 1. Direct Memory Decode (Zero Disk I/O)
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("OpenCV failed to decode image.")

        model = ml_state["model"]
        
        # 2. Forward Pass Inference
        results = model.predict(
            source=img,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        # 3. Payload Extraction
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

        logger.info(f"Analyzed '{file.filename}': Found {len(detections)} objects.")

        return DetectionResponse(
            filename=file.filename,
            total_detections=len(detections),
            detections=detections
        )

    except Exception as e:
        logger.error(f"Inference Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during detection.")