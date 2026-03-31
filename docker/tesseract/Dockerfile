FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/tesseract/requirements.txt /tmp/tess-requirements.txt
RUN pip install --no-cache-dir -r /tmp/tess-requirements.txt

COPY docker/tesseract/run.py /app/run.py

ENTRYPOINT ["python", "/app/run.py"]
