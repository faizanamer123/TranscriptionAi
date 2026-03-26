# Use Python 3.11 with CUDA support if needed, but slim is better for CPU-only/General
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg is required for faster-whisper to handle various media formats
# build-essential might be needed for some python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create uploads directory
RUN mkdir -p uploads && chmod 777 uploads

# Expose the application port
EXPOSE 8001

# Environment variables for the app
ENV PORT=8001
ENV MAX_FILE_SIZE=524288000

# Run the application
CMD ["python1.py"]
