FROM python:3.12-slim

WORKDIR /app

# Core deps + PaddleOCR (for PP-OCRv6 small model)
RUN pip install --no-cache-dir \
    uvicorn fastapi python-multipart httpx \
    onnxruntime \
    Pillow numpy \
    huggingface_hub \
    paddleocr paddlepaddle

# Pre-download the OCR model during build so startup is instant
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)"

COPY ocr_server.py .

EXPOSE 5004

CMD ["uvicorn", "ocr_server.py:app", "--host", "0.0.0.0", "--port", "5004"]
