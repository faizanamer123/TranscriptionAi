# TranscribeAI Enterprise API

High-performance parallel transcription system powered by FastAPI and Faster-Whisper.

## Features
- Parallel media processing
- Hardware-aware scaling (CPU/GPU)
  - Automatically detects CPU cores and GPU availability
  - Dynamically calculates optimal worker counts based on available RAM (1.2GB per worker)
  - Distributes load across multiple GPUs if available
- Real-time job status and progress
- Secure API Key authentication
- Dynamic resource monitoring

## Getting Started
The frontend is available at:
[http://localhost:8001](http://localhost:8001) or [http://localhost:8002](http://localhost:8002) (depending on the run method)

### Running the Project

#### Option A: Uvicorn (Recommended for Development)
Run the server on port 8001 with auto-reload:
```bash
python -m uvicorn python1:app --host 0.0.0.0 --port 8001 --reload
```

#### Option B: Python Script (Default)
Run the server on port 8002:
```bash
python python1.py
```

### Access the Dashboard
Open the corresponding URL in your browser:
- If using **Option A**: [http://localhost:8001](http://localhost:8001)
- If using **Option B**: [http://localhost:8002](http://localhost:8002)

## Sharing Access
Once the server is running, you can share it with others:

### 1. Local Network (WiFi/LAN)
Share with anyone on the same network using your IP address:
- **Find IP**: Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux).
- **URL**: `http://<YOUR_IP_ADDRESS>:8001` (if using uvicorn)

### 2. Public Access (Tunneling)
Share with someone anywhere in the world using a tunnel:
- **Quick Access**: `ssh -p 443 -R0:localhost:8002 a.pinggy.io`
- This provides a public URL (e.g., `https://random-name.pinggy.link`).

### 3. Cloud Deployment
You can deploy this project to services like **Render**, **Railway**, or any VPS (Virtual Private Server) that supports Python.