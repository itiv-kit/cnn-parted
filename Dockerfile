# Dockerfile for CNNParted
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install the project using uv
RUN uv venv --python 3.11 && \
    . .venv/bin/activate && \
    uv pip install -e ".[nvidia]"

# Set up environment
ENV SIM_ROOT_DIR=/app
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["bash"]
