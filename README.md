---
title: My Transcription Api
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# TranscribeAI Enterprise API

High-performance parallel transcription system powered by FastAPI and Faster-Whisper.

## Features
- Parallel media processing
- Hardware-aware scaling (CPU/GPU)
  - Automatically detects CPU cores and GPU availability
  - Dynamically calculates optimal worker counts based on available RAM (1.5GB per worker)
  - Distributes load across multiple GPUs if available
- Real-time job status and progress
- Secure API Key authentication
- Dynamic resource monitoring

## Getting Started
The frontend is available at:
[http://localhost:8002](http://localhost:8002)

Simply open this URL in your browser to access the TranscribeAI interface.

### Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `python python1.py`
3. Access the dashboard at `http://localhost:8002`

## Docker Usage
To run the application using Docker:

1. Build the image:
   ```bash
   docker build -t transcribe-ai .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8002:8002 --name transcribe-ai transcribe-ai
   ```

3. Open `http://localhost:8002` in your browser.

The Docker container is pre-configured to handle hardware-aware scaling and will automatically detect the available resources in your container environment.