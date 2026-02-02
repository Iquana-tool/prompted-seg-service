# 1. Choose the base image via ARG
# Default to CUDA
ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
FROM $BASE_IMAGE AS base

# --- Stage 1: Builder ---
FROM base AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python & build tools IF we are on an NVIDIA (Ubuntu) image
# (The python-slim image already has these, but nvidia-base doesn't)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Configure git to use token for private repo access
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Sync dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- Stage 2: Final Runtime ---
FROM base AS runtime

# Copy uv from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up environment
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Install runtime system dependencies (GL, etc.)
# We also ensure Python is present for the NVIDIA base
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configure git to use token for private repo access in runtime
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
    cd /tmp && \
    echo "https://${GITHUB_TOKEN}@github.com" > /root/.git-credentials && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global credential.helper store && \
    GIT_CONFIG_GLOBAL=/root/.gitconfig git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

# Copy the venv from builder
COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8001

CMD ["uv", "run", "--upgrade", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
