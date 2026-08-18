FROM python:3.12-slim

WORKDIR /app

# Core deps only — lightweight, no PaddlePaddle
RUN pip install --no-cache-dir \
    uvicorn fastapi python-multipart httpx \
    onnxruntime \
    Pillow numpy \
    huggingface_hub

COPY ocr_server.py .

EXPOSE 5004

CMD ["uvicorn", "ocr_server.py:app", "--host", "0.0.0.0", "--port", "5004"]
