# TranscriptionAi

Local web service that **transcribes** audio/video (WhisperX), **aligns** words, optionally **diarizes speakers** (pyannote via Hugging Face), and writes **markdown conversation transcripts**. A **FastAPI** server queues jobs; **worker processes** run the models and report progress for the bundled **HTML UI**.

## What it does

| Step | Behavior |
|------|----------|
| Upload | Accepts common formats (e.g. MP3, WAV, M4A, MP4, MOV, MKV) into `uploads/`. |
| Short files (≤ ~10 min) | Single-pass transcribe → align → diarize (if token set). |
| Long files (> ~10 min) | Overlapping chunks for ASR; timestamps merged; **full-file** diarization for consistent speaker labels. |
| Output | Markdown under `transcriptions/` with speaker lines and time ranges. |
| Control | Pause/resume queue, cancel jobs, optional CPU/memory–based worker scaling. |

**Diarization** needs `HF_TOKEN` (see [Hugging Face access](https://huggingface.co/settings/tokens) and accept pyannote model terms). Without it, transcription still runs but **speaker labels are disabled**.

## Project layout

| File | Role |
|------|------|
| `python1.py` | Entry: runs `server.run()` (uvicorn). |
| `server.py` | FastAPI app, routes, job list, dispatcher, `asyncio` lock for shared state. |
| `worker.py` | Pool workers: WhisperX + alignment + diarization pipeline. |
| `transcript.py` | Post-processing: UNKNOWN fill-in, merge by speaker, orphan cleanup, markdown. |
| `chunking.py` | Long-audio chunking and speaker count hints. |
| `settings.py` | Env, logging, `HF_TOKEN`, constants (`JOBS_MAX`, chunk length/overlap, sample rate). |
| `hardware.py` | CPU/RAM–based worker limits. |
| `index.html` | Static UI (Tailwind CDN). |
| `requirements.txt` | Pinned Python dependencies. |
| `run.sh` | Runs `.venv/bin/python python1.py` (macOS/Linux if `python` is missing). |

## Requirements

### Software

- **Python 3.11** (recommended). WhisperX **3.8.x** does not support Python **3.14+**; use 3.11 in a venv.
- **FFmpeg** on `PATH` (pydub / decoding).  
  - macOS: `brew install ffmpeg`  
  - Ubuntu: `sudo apt install ffmpeg`  
  - Windows: install FFmpeg and add its `bin` to PATH.
- **Git** (optional, for clone).

### Python packages

Install everything with:

```bash
pip install -r requirements.txt
```

Main stack (see `requirements.txt` for versions):

- **Transcription / ML:** `whisperx`, `faster-whisper`, `torch`, `torchaudio`, `transformers`
- **Diarization:** `pyannote.audio` (+ related `pyannote.*`)
- **Audio:** `pydub`, `librosa`, `soundfile`, `ffmpeg-python`
- **Web:** `fastapi`, `uvicorn`, `python-multipart`
- **Utils:** `python-dotenv`, `psutil`, `numpy`, `pandas`, etc.

### Hardware

- **Minimum:** ~4 GB RAM, 2 cores (CPU-only is slow for large-v3).
- **Recommended:** 8 GB+ RAM, NVIDIA GPU with CUDA for reasonable speed on `large-v3`.
- **Disk:** several GB for models and caches (Hugging Face / PyTorch).

### Environment variable

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | For diarization | Hugging Face token with access to pyannote models. |

Copy `.env.example` to `.env` and set `HF_TOKEN=...`.

Tunable constants (code, not env): `JOBS_MAX`, `CHUNK_SEC`, `CHUNK_OVERLAP_SEC`, `LONG_AUDIO_SEC`, `SAMPLE_RATE` in `settings.py`. **HTTP port** and bind address are in `server.py` (`run()` defaults to `0.0.0.0:8002`).

## Installation

1. **Clone** the repo and `cd` into it.

2. **Create a virtual environment** (Python 3.11):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure `.env`:**

   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=...
   ```

## How to run

From the project root (with venv activated):

```bash
python3 python1.py
```

If your shell has no `python` command (common on macOS), use:

```bash
.venv/bin/python python1.py
```

Or:

```bash
chmod +x run.sh
./run.sh
```

Open **http://localhost:8002** (or **http://127.0.0.1:8002**).

**CORS** is restricted to **`http://localhost` / `http://127.0.0.1`** (any port). Browsing the API from another host name (e.g. LAN IP) requires changing `allow_origin_regex` / `allow_origins` in `server.py`.

## API (summary)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `index.html`. |
| `POST` | `/upload` | Multipart file upload; creates jobs (`waiting`). |
| `GET` | `/status` | Queue/worker stats and recent jobs (newest first). |
| `POST` | `/pause` | Pause global processing; marks running jobs paused. |
| `POST` | `/resume` | Resume. |
| `POST` | `/cancel/{job_id}` | Cancel if job is in a cancellable state; otherwise **400**. |
| `DELETE` | `/clear` | Kill workers, clear job list (destructive). |
| `GET` | `/download/{filename}` | Download a `.md` file from `transcriptions/` (safe basename only). |

## Output format

Markdown transcripts look like:

```markdown
# Speaker Diarized Conversation

**Detected 2 speakers**

**SPEAKER_00 (0:00.00-0:04.04)**: Example line one.

**SPEAKER_01 (0:04.92-0:08.00)**: Example line two.
```

## Troubleshooting

- **`python`: command not found** — Use `python3` or `.venv/bin/python`.
- **Diarization disabled / warnings** — Set `HF_TOKEN`; accept model conditions on Hugging Face.
- **FFmpeg / torchcodec warnings** — Install FFmpeg; on Apple Silicon, mismatched `torchcodec` vs PyTorch may warn; audio often still works via pydub/whisperx paths.
- **CUDA not used** — Install a CUDA build of PyTorch matching your driver; CPU fallback works but is slower.
- **Out of memory** — Close other apps; fewer concurrent jobs; shorter files or smaller Whisper model (would require code change from `large-v3`).

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX), [OpenAI Whisper](https://github.com/openai/whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/).
