FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uvicorn fastapi python-multipart httpx 'markitdown[all]'

COPY markitdown_server.py .

EXPOSE 5003

CMD ["uvicorn", "markitdown_server:app", "--host", "0.0.0.0", "--port", "5003"]
