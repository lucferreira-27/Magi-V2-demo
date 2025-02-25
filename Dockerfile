FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directories for input/output
RUN mkdir -p /data/test_assets/pages /data/test_assets/characters /data/output /app/logs

# Set environment variables
ENV TEST_DIR=/data/test_assets
ENV OUTPUT_DIR=/data/output

# Run the application
ENTRYPOINT ["python", "main.py"] 