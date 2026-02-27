FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY docker/pool_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pool dispatcher library from repo root
COPY claude_cli_pool ./claude_cli_pool

# Copy pool service entrypoint
COPY docker/pool_service/pool_service.py .

EXPOSE 8000
CMD ["uvicorn", "pool_service:app", "--host", "0.0.0.0", "--port", "8000"]
