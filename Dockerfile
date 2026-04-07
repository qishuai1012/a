# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Install additional dependencies
RUN pip install --no-cache-dir "uvicorn[standard]" gunicorn

# Copy the rest of the application
COPY . .

# Expose default port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command - dynamically determine service type based on environment variable
CMD ["bash", "-c", "if [ -z \"$SERVICE_NAME\" ]; then python main.py server --host 0.0.0.0 --port 8000; else python microservice_main.py $SERVICE_NAME --host 0.0.0.0 --port $PORT; fi"]