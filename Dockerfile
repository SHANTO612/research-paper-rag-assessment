FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

 # Install Python dependencies first for better caching
COPY requirements.txt ./requirements.txt

# Increase pip timeout to avoid failures downloading large wheels
ENV PIP_DEFAULT_TIMEOUT=1200

# Install CPU-only PyTorch first (prevents pulling large CUDA wheels), then the rest of requirements.
# Using --prefer-binary helps pip pick prebuilt wheels when available.
RUN pip install --no-cache-dir --prefer-binary -f https://download.pytorch.org/whl/cpu/torch_stable.html torch==2.9.0 \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

 # Copy application code
COPY . /app/

 # Create persistent upload directory
RUN mkdir -p /app/uploaded_papers

EXPOSE 8000 8501

ENV UPLOAD_DIR=/app/uploaded_papers \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/research_papers

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


