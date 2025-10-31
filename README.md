# Research Paper RAG System – Developer Guide

This README focuses on running and operating this repository (without changing the original assessment brief in `README.md`).

## Quick Start

- Prerequisites
  - Docker Desktop (recommended)
  - Python 3.10+ (for local runs without Docker)

- One command (Docker Compose)
```bash
# In repo root
docker compose up -d --build
# API:       http://localhost:8000
# Streamlit: http://localhost:8501
```

- View logs (optional)
```bash
docker compose logs -f api
docker compose logs -f streamlit
```

- Stop stack
```bash
docker compose down
```

## Environment Configuration

Create `.env` (copy from `.envExample`) and set values as needed. Common keys:
```
# Database (Neon/Postgres, async driver)
DATABASE_URL=postgresql+asyncpg://<user>:<pass>@<host>/<db>?sslmode=require

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# LLM selection: gemini 
LLM_TYPE=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash


# Embeddings (match your chosen model’s vector size)
VECTOR_SIZE=768
```

Notes:
- The API container uses `DATABASE_URL` and will auto-create tables on startup.
- The Streamlit container talks to the API at `http://api:8000/api` (service name inside Docker network).

## Running Locally (without Docker)

- Create and activate venv, install deps
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Set environment, start API
```powershell
$env:DATABASE_URL = "postgresql+asyncpg://<user>:<pass>@<host>/<db>?sslmode=require"
$env:QDRANT_HOST = "localhost"
$env:QDRANT_PORT = "6333"
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

- Start Qdrant locally (if not using Docker Compose)
```powershell
docker run -d --name rag-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.11.0
```

- Run Streamlit against local API
```powershell
.\venv\Scripts\Activate.ps1
$env:API_BASE = "http://127.0.0.1:8000/api"
python -m streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
Copy URL: http://localhost:8501
```
## Key Endpoints

- Health: `GET /`
- Papers:
  - `POST /api/papers/upload`
  - `GET /api/papers`
  - `GET /api/papers/{paper_id}`
  - `DELETE /api/papers/{paper_id}`
- Query:
  - `POST /api/query?query=...&top_k=5[&paper_ids=1&paper_ids=2]`
  - `GET /api/query/history`
  - `POST /api/query/history/{query_id}/rate?rating=1..5`
  - `GET /api/query/analytics/popular`

## Common Troubleshooting

- Streamlit shows “Backend error … connection refused”
  - Ensure API is healthy: `curl http://localhost:8000/`
  - In Docker, verify service health: `docker compose ps`, `docker compose logs -f api`

- Qdrant connection errors
  - In Docker: `docker compose up -d qdrant`
  - Outside Docker: run the `docker run qdrant` command above

- Neon connection errors
  - Confirm `DATABASE_URL` uses the async driver `postgresql+asyncpg://...` and includes `sslmode=require`

## Notes on Embeddings

- Default embedding model in code is Sentence-Transformers `all-mpnet-base-v2` (768-d vectors). Set `VECTOR_SIZE=768` unless you change the model.

## Licensing / Contributions

- Use this guide to run and operate the project.
- For submissions, follow the original `README.md` brief and `SUBMISSION_GUIDE.md`.
