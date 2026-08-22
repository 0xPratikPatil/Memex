FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# System deps for OpenCV (libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps via uv (RapidOCR = PaddleOCR models without PaddlePaddle)
COPY ocr_server.py .
RUN uv pip install --system --compile-bytecode \
    uvicorn fastapi python-multipart httpx \
    rapidocr-onnxruntime \
    pypdfium2 \
    Pillow numpy

EXPOSE 5004

CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "5004"]
