# =============================================================================
# Multi-Stage Docker Build for House Price Prediction API
# =============================================================================
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# Stage 2: Production
FROM python:3.11-slim
WORKDIR /app
# Create non-root user for security
RUN groupadd -r mlapp && useradd -r -g mlapp mlapp
# Copy only necessary artifacts from builder
COPY --from=builder /root/.local /home/mlapp/.local
ENV PATH=/home/mlapp/.local/bin:$PATH
# Copy application code
COPY main.py .
COPY frontend.html ./static/
COPY model_registry/ ./model_registry/
# Set proper permissions
RUN chown -R mlapp:mlapp /app
USER mlapp
# Expose API port
EXPOSE 8000
# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
