FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make Python scripts executable
RUN chmod +x indexer.py api.py watcher.py example_client.py

# Create volume mount points
RUN mkdir -p /app/chroma_index

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose FastAPI port
EXPOSE 8000

# Default command - can be overridden in docker-compose
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"] 