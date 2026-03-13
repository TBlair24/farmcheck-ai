FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/    ./api/
COPY models/ ./models/
COPY runs/   ./runs/

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]