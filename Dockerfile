# Use Python 3.12 slim image (lightweight)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for your ML libraries)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install PyTorch CPU-only first with the special index
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch/torchvision since already installed)
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy application code
COPY . .

# Cloud Run requires the app to listen on 0.0.0.0 and port 8080
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]