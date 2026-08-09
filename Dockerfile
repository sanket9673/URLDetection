# Stage 1: Build stage for installing dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU first to avoid heavy GPU image dependency size
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.0

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies (LightGBM requires libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000
