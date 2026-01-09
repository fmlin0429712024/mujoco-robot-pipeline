

FROM ros:humble-ros-base

# Install system dependencies for MuJoCo and OpenGL (EGL/OSMesa)
# libgl1-mesa-dev, libgl1-mesa-glx, libosmesa6-dev are standard for headless rendering
# libglew-dev might be needed
# git, build-essential for compiling some python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    ffmpeg \
    unzip \
    wget \
    wget \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for MuJoCo to use EGL (Hardware acceleration if avail) or OSMesa (Software)
# Cloud Run Gen 2 supports roughly standard linux environment.
# Typically for pure CPU rendering (software), default mujoco might just work or need MUJOCO_GL=osmesa
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies from pyproject.toml
# We need to install `.` to install the package itself + `streamlit`
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir streamlit

# Install Triton client for inference
RUN pip install --no-cache-dir tritonclient[grpc] geventhttpclient

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose the port
EXPOSE 8080

# Run the application
CMD ["streamlit", "run", "app.py"]
