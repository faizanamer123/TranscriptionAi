# 🎯 Transcription AI - Advanced Audio Transcription System

A powerful, production-ready audio transcription system with speaker diarization, real-time job management, and conversation formatting.

## ✨ Features

### 🎙️ **Core Transcription**
- **Multi-format support** - MP3, WAV, M4A, and more
- **Speaker diarization** - Automatically identifies different speakers
- **Conversation formatting** - Clean, readable transcript output
- **High accuracy** - Powered by WhisperX and advanced AI models

### 🚀 **Advanced Features**
- **Real-time job management** - Track transcription progress
- **Pause/Resume functionality** - Control transcription processing
- **Cancel support** - Stop jobs instantly with proper cleanup
- **Batch processing** - Handle multiple files simultaneously
- **Chunk processing** - Efficient handling of long audio files

### 🎨 **User Interface**
- **Modern web interface** - Clean, responsive design
- **Real-time updates** - Live progress tracking
- **Instant feedback** - Optimistic UI updates
- **Professional output** - Formatted conversation transcripts

### ⚡ **Performance**
- **GPU acceleration** - CUDA support for faster processing
- **Parallel processing** - Multi-core CPU utilization
- **Memory optimization** - Efficient resource management
- **Chunked processing** - Handle files of any length
- **Hardware-aware scaling** - Automatically detects CPU cores and GPU availability
- **Dynamic resource monitoring** - Real-time system resource tracking

## 🛠️ Installation

### Prerequisites
- **Python 3.10 or 3.11** (Highly Recommended). Python 3.13 is NOT supported due to the removal of the `audioop` module required by `pydub`.
- **FFmpeg**: Required for speaker diarization and audio processing
- **CUDA-compatible GPU** (recommended for performance)

### Quick Setup

1. **Install FFmpeg (Windows)**
   ```bash
   # Download the "full-shared" build from gyan.dev
   # https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z
   # Extract and add the bin folder to your PATH
   # Verify: ffmpeg -version
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set up Environment**
   ```bash
   # Create .env file with your Hugging Face token
   echo "HF_TOKEN=your_huggingface_token_here" > .env
   ```

5. **Run the Application**
   ```bash
   python python1.py
   ```

6. **Access Web Interface**
   Open `http://localhost:8000` in your browser

## 📋 Requirements

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores, NVIDIA GPU
- **Storage**: 1GB+ free space for models and temp files

### Software Requirements
- Python 3.10 or 3.11
- CUDA Toolkit 11.0+ (for GPU acceleration)
- FFmpeg (for audio processing)

## 🎯 Usage

### Web Interface
1. **Upload audio files** - Drag & drop or click to select
2. **Monitor progress** - Real-time job status updates
3. **Control jobs** - Pause, resume, or cancel transcriptions
4. **Download results** - Get formatted conversation transcripts

### Conversation Transcript Output
The system generates clean conversation transcripts:

```markdown
# Speaker Diarized Conversation

**Detected 2 speakers**

**SPEAKER_00 (0:00.03-0:04.04)**: Good morning. I am Daniel. I am the head of this company.

**SPEAKER_01 (0:04.92-0:08.00)**: Nice to meet you, sir. I am Simon.
```


## 🔧 Configuration

### Environment Variables
Create a `.env` file:

```env
# Hugging Face token for speaker diarization
HF_TOKEN=your_huggingface_token_here

# Optional: CUDA device
CUDA_DEVICE=0

# Optional: Worker count
MAX_WORKERS=4
```

### Advanced Settings
- **Worker configuration** - Automatically optimized for your hardware
- **Chunk duration** - Adjustable for different audio lengths (default: 5 minutes)
- **Speaker detection** - Configurable sensitivity settings
- **Memory management** - 1.2GB per worker allocation

## 🚀 Performance Features

### Hardware Optimization
- **Automatic detection** - CPU cores, GPU availability, RAM
- **Dynamic scaling** - Optimal worker count calculation
- **Multi-GPU support** - Distribute across available GPUs
- **Memory monitoring** - Real-time resource tracking

### Processing Features
- **Chunk processing** - Handle files of any length
- **Parallel workers** - Multi-core utilization
- **GPU acceleration** - CUDA support for faster processing
- **Smart cancellation** - Proper cleanup and resource management

## 📊 API Endpoints

### Job Management
- `POST /upload` - Upload audio files
- `GET /status` - Get all job statuses
- `POST /pause` - Pause all processing
- `POST /resume` - Resume processing
- `POST /cancel/{job_id}` - Cancel specific job
- `DELETE /clear` - Clear all jobs

### File Operations
- `GET /download/{job_id}` - Download transcription result
- `GET /files/{filename}` - Access generated files

## 🔍 Troubleshooting

### Common Issues

**CUDA not available**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**FFmpeg not found**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - Download from https://ffmpeg.org/download.html
# Use "full-shared" build for AI libraries
```

**Memory issues**
- Reduce `MAX_WORKERS` in environment
- Use smaller chunk sizes
- Close other applications

**Slow processing**
- Ensure GPU acceleration is enabled
- Check system resources
- Optimize audio file quality

### Debug Mode
Enable debug logging:
```bash
export PYTHONPATH=$PYTHONPATH:.
python python1.py
```

## 🌐 Network Access

### Local Network
Share with anyone on the same network:
```bash
# Find your IP
ipconfig  # Windows
ifconfig  # Mac/Linux

# Access via: http://<YOUR_IP>:8000
```

### Cloud Deployment
Deploy to services like:
- **Render**
- **Railway** 
- **VPS** (Virtual Private Server)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **OpenAI Whisper** - Base transcription model
- **WhisperX** - Enhanced transcription with alignment
- **PyAnnote.audio** - Speaker diarization
- **FastAPI** - Web framework
- **PyTorch** - Deep learning framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

---

**🎯 Transcription AI** - Professional audio transcription made simple