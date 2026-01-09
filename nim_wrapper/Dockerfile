FROM ros:humble-ros-base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    tritonclient[grpc] \
    numpy \
    torch \
    httpx

# Copy application code
COPY . /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-u", "main.py"]
