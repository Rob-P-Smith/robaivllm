FROM python:3.12-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink so robaivenv python symlinks work
RUN ln -s /usr/local/bin/python3 /usr/bin/python3

# Copy application code
COPY robaivllm/*.py ./

# Create non-root user for security
RUN useradd -m -u 1000 proxyuser && \
    mkdir -p /app/data && \
    chown -R proxyuser:proxyuser /app

# Switch to non-root user
USER proxyuser

# Set environment variables
# venv will be mounted at /venv, prepend to PATH
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Run proxy server
CMD ["python3", "thinking_proxy.py"]
