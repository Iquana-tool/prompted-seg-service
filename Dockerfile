# Use a lightweight Python base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (git + any required libraries)
RUN apt-get update --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed for dependency installation
COPY . .

# Sync dependencies using uv
RUN uv sync --no-cache

# Install torch with or without CUDA
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Clone and install sam2 from GitHub
RUN git clone https://github.com/facebookresearch/sam2.git && \
    cd sam2 && \
    SAM2_BUILD_CUDA=0 uv pip install . --no-cache-dir && \
    cd .. && rm -rf sam2

# Install sam3
RUN git clone https://github.com/facebookresearch/sam3.git && \
    cd sam3 && \
    uv pip install -e ".[train,dev]" && \
    cd ..

# Expose the FastAPI port (default: 8000)
EXPOSE 8000

RUN uv pip freeze

# Run the FastAPI app using uvicorn
CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "8000"]
