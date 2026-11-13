FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Tesseract supports the application's optional PDF OCR path; the Vietnamese
# language package matches the default `vie+eng` configuration.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY api ./api
COPY app ./app
COPY configs ./configs
COPY src ./src
COPY scripts ./scripts
COPY data/eval ./data/eval

RUN mkdir -p data/raw/uploads data/processed

EXPOSE 8000 8501
