# Use the official Python slim image (lightweight)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libpq-dev is often required for PostgreSQL Python adapters (psycopg2/pg8000)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY run.py .

# Production Environment Variables
ENV FLASK_DEBUG=0
# Cloud Run expects the app to listen on the port defined by the PORT env var
ENV PORT=8080

# Run Gunicorn
# --workers 1: Optimal for 1 vCPU (avoids context switching overhead)
# --threads 8: Handles concurrent I/O (DB queries, GCS uploads) efficiently
# --timeout 0: Let Cloud Run manage the request timeout
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 run:app