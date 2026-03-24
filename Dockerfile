# 1. Base Image: Lightweight Debian Python
FROM mirror.gcr.io/library/python:3.10-slim

# 2. System Variables & Python Configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- YOLOv8 SPECIFIC CONFIGURATION ---
ENV CONF_THRESHOLD=0.40
ENV IOU_THRESHOLD=0.50

# --- OS-LEVEL DEPENDENCIES PATCH ---
# Inject OpenGL and GLib C++ libraries required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory
WORKDIR /workspace

# 4. Layer Caching: Install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Inject Microservice Code
COPY app/ app/
COPY models/ models/

# 6. Expose the API Port
EXPOSE 8000

# 7. Execute the Engine
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]