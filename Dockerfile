# --- stage 1: build the Astro frontend ---
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- stage 2: python runtime ---
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY wire/ ./wire/
COPY api.py run.py benchmark.py ./
COPY data/ ./data/
COPY seed/ ./seed/
COPY --from=web /web/dist ./frontend/dist

ENV PORT=2424
EXPOSE 2424
CMD ["python", "api.py"]
