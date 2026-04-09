# DefectDetector Service Dockerfile
# Multi-stage build for minimal production image

# =============================================================================
# Stage 1: Build stage - install dependencies
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# PyTorch CPU-only wheels (default PyPI resolves to CUDA builds on Linux).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Production stage - minimal runtime image
# =============================================================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVICE_PORT=8400 \
    SERVICE_HOST=0.0.0.0

WORKDIR /app

# Install runtime dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -ms /bin/bash appuser

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code and weights from this repo (see models/defect_cls.pt or MODEL_PATH)
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser models/ /app/models/

# Switch to non-root user
USER appuser

# Expose service port (documentation)
EXPOSE ${SERVICE_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('SERVICE_PORT','8400'); urllib.request.urlopen('http://127.0.0.1:%s/health' % p, timeout=3)"

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host ${SERVICE_HOST} --port ${SERVICE_PORT}"]

