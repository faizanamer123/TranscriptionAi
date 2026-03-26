# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
# ffmpeg is required for faster-whisper to handle various media formats
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Set environment variables for production
ENV PORT=7860
ENV API_KEY=dev-key-123
ENV ALLOWED_ORIGINS=*

# Command to run the application
CMD ["python", "python1.py"]
