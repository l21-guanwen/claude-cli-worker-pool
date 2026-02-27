FROM python:3.11-slim

# Install Node.js (required for claude CLI)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean

# Install claude CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Install Python deps for worker API
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker_api.py .
COPY start_workers.sh .
# Strip Windows CRLF line endings and make executable
RUN sed -i 's/\r//' start_workers.sh && chmod +x start_workers.sh

# Expose base port and the next 9 ports to support up to 10 workers per account.
# WORKERS_PER_ACCOUNT controls how many are actually started at runtime.
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009

CMD ["./start_workers.sh"]
