# ── Stage 1: Compile TypeScript ───────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /build
COPY package.json package-lock.json tsconfig.json ./
RUN npm ci --only=dev
COPY frontend/ frontend/
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend source
COPY backend/ backend/
COPY __init__.py .

# Data files referenced at runtime
COPY parameters.yaml .
COPY src/data_retrieval/ src/data_retrieval/

# Frontend static assets
COPY frontend/app.html frontend/app.html
COPY frontend/styles/  frontend/styles/
COPY --from=frontend-builder /build/frontend/dist/ frontend/dist/

EXPOSE 8080
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
