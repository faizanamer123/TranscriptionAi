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

### Diarization stages (what runs under the hood)

| Stage | Role in this project |
|-------|----------------------|
| **VAD / speech regions** | Pyannote’s **segmentation** model localizes speech (neural “who speaks when” frames). Optional **Silero VAD** (`DIARIZATION_SILERO_VAD=1`) can mask non-speech before diarization to cut false alarms in silence. |
| **Speaker embeddings** | Pipeline **embedding** model (e.g. ECAPA-style weights bundled with the HF checkpoint). Optional `DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP=1` uses cleaner frames when overlaps confuse the extractor. |
| **Clustering** | **VBx** (default on community / 3.1 checkpoints) uses **agglomerative hierarchical clustering** with a **cosine**-related metric, then variational refinement; tune with `DIARIZATION_CLUSTERING_THRESHOLD`, `DIARIZATION_VBX_FA`, `DIARIZATION_VBX_FB` when you know your data. |
| **Boundaries** | Pyannote produces segment boundaries; optional **`DIARIZATION_RESEGMENT_SMOOTH=1`** applies extra gap/island cleanup on the diarization table before word assignment. |

## Project layout

| File | Role |
|------|------|
| `python1.py` | Entry: runs `server.run()` (uvicorn). |
| `server.py` | FastAPI app, routes, job list, dispatcher, `asyncio` lock for shared state. |
| `worker.py` | Pool workers: WhisperX + alignment + diarization pipeline. |
| `diarization.py` | Pyannote load/tuning, optional Silero VAD, diarization-table smoothing → WhisperX assignment. |
| `transcript.py` | UNKNOWN fill, word smoothing, island collapse, speaker merge, **temporal line glue**, orphans → markdown. |
| `chunking.py` | Long-audio chunking (overlap for ASR). |
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

### Environment variables

Copy **`.env.example`** to **`.env`**, set **`HF_TOKEN`**, then uncomment and adjust any optional lines. Defaults for anything omitted are in **`settings.py`**.

| Variable | Purpose |
|----------|---------|
| **`HF_TOKEN`** | **Required for diarization.** [Hugging Face token](https://huggingface.co/settings/tokens); accept **pyannote/speaker-diarization-3.1** (and **segmentation-3.0** if prompted). |

**Models and speaker bounds** (optional)

| Variable | Purpose |
|----------|---------|
| `DIARIZATION_MODEL` | HF pipeline id (default `pyannote/speaker-diarization-3.1`). |
| `DIARIZATION_FALLBACK_MODEL` | Used if the primary model fails to load (default `pyannote/speaker-diarization-community-1`). |
| `DIARIZATION_MIN_SPEAKERS` | Minimum speakers pyannote may return (default `1`). |
| `DIARIZATION_MAX_SPEAKERS` | Upper bound for clustering (default `30`). Set high enough for **casts** (e.g. animation with many voices); a low value (like `5`) can merge distinct characters. Use `none` for no cap. |

**Silero VAD** (optional)

| Variable | Purpose |
|----------|---------|
| `DIARIZATION_SILERO_VAD` | Set `1` to mask non-speech before diarization (downloads model via `torch.hub` once). Default **off**. |
| `DIARIZATION_SILERO_PAD_MS` | Padding around Silero speech segments (default `40`). |

**Diarization table smoothing** (optional)

| Variable | Purpose |
|----------|---------|
| `DIARIZATION_RESEGMENT_SMOOTH` | Merge small gaps / short islands on the diarization dataframe before word assignment. Default **on**; set `0` to disable. |
| `DIARIZATION_RESEGMENT_MERGE_GAP_SEC` | Max gap (seconds) to merge adjacent same-speaker rows when smoothing is on. |
| `DIARIZATION_RESEGMENT_ISLAND_SEC` | Max duration of a middle “island” to collapse when smoothing is on. |

**Embeddings** (optional)

| Variable | Purpose |
|----------|---------|
| `DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP` | Set `1` to favor non-overlapping frames for speaker embeddings when the pipeline supports it. Default **off**. |

**Pyannote hyperparameters** (optional; only if the loaded pipeline exposes the attribute)

| Variable | Purpose |
|----------|---------|
| `DIARIZATION_SEGMENTATION_THRESHOLD` | Binarization threshold for non-powerset segmentation (higher → stricter “speech on”). |
| `DIARIZATION_MIN_DURATION_OFF` | Minimum silence duration (seconds) for segmentation. |
| `DIARIZATION_CLUSTERING_THRESHOLD` | Clustering distance threshold (lower → more speaker splits). |
| `DIARIZATION_VBX_FA` / `DIARIZATION_VBX_FB` | VBx refinement parameters when the pipeline uses VBx. |

**Transcript post-processing** (optional)

| Variable | Purpose |
|----------|---------|
| `TRANSCRIPT_RESOLVE_UNKNOWN` | Set `0` to skip filling UNKNOWN word speakers (default **on**: time-window vote / nearest label). |
| `TRANSCRIPT_RESOLVE_WINDOW_SEC` | Half-width of the time window (seconds) for UNKNOWN resolution. |
| `TRANSCRIPT_RESOLVE_MAX_PASSES` | Iteration cap for UNKNOWN resolution. |
| `TRANSCRIPT_MERGE_MAX_GAP_SEC` | Max gap (seconds) between words to keep the same speaker on one markdown line. |
| `TRANSCRIPT_MERGE_ALLOW_MICRO_FLIP` | Set `1` to allow brief cross-speaker absorption of tiny runs (see `TRANSCRIPT_MERGE_MICRO_FLIP_SEC` in `settings.py`). |
| `TRANSCRIPT_SMOOTH_SPEAKERS` | Sliding-window majority on word-level speakers. Default **on**; set `0` to disable. |
| `TRANSCRIPT_SMOOTH_RADIUS` | Neighbor radius (words) for smoothing. |
| `TRANSCRIPT_SMOOTH_PASSES` | Passes for the first smoothing stage (a lighter pass runs after island collapse). |
| `TRANSCRIPT_COLLAPSE_ISLANDS` | Collapse short A→B→A word runs. Default **on**; set `0` to disable. |
| `TRANSCRIPT_ISLAND_MAX_SEC` | Max duration (seconds) of an island to treat as a flicker. |
| `TRANSCRIPT_CLEAN_ORPHANS` | Relabel short / UNKNOWN segments between the same speaker. Default **on**; set `0` to disable. |
| `TRANSCRIPT_ORPHAN_MIN_DURATION_SEC` | Duration threshold (seconds) for orphan cleanup. |
| `TRANSCRIPT_TEMPORAL_GLUE` | Merge adjacent markdown lines when **gaps + segment lengths** look like diarization jitter (no punctuation rules). Default **on**. |
| `TRANSCRIPT_TEMPORAL_GLUE_MAX_GAP_SEC` | Max inter-line gap (seconds) to consider for glue. |
| `TRANSCRIPT_TEMPORAL_GLUE_ULTRA_GAP_SEC` | Always merge if gap ≤ this (boundary jitter). |
| `TRANSCRIPT_TEMPORAL_GLUE_SHORT_SEC` | “Short side” duration (seconds) for asymmetric merges. |
| `TRANSCRIPT_TEMPORAL_GLUE_LONG_CAP_SEC` | Avoid merging two long turns unless the gap is tiny. |

**Other tuning:** `JOBS_MAX`, `CHUNK_SEC`, `CHUNK_OVERLAP_SEC`, `LONG_AUDIO_SEC`, `SAMPLE_RATE`, and `TRANSCRIPT_MERGE_MICRO_FLIP_SEC` live in **`settings.py`** if not overridden by env. **HTTP** bind and port are set in **`server.py`** (`run()` defaults to `0.0.0.0:8002`).

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

## Diarization & transcript accuracy

Typical failure modes in the markdown output:

| Symptom | Likely cause | What to try |
|--------|----------------|-------------|
| One sentence split across **SPEAKER_00 / SPEAKER_01** with **tiny time gaps** | Diarization flip mid-utterance | Defaults now include **temporal glue**, **word smoothing**, and **island collapse**. Tune `TRANSCRIPT_TEMPORAL_GLUE_*` or raise `TRANSCRIPT_MERGE_MAX_GAP_SEC` slightly. |
| **Two people** in one line (e.g. question + answer) | Same speaker label on adjacent words | **Smoothing** + **collapse**; lower `DIARIZATION_CLUSTERING_THRESHOLD` slightly for more splits, or raise `TRANSCRIPT_SMOOTH_RADIUS`. |
| **Animation / many characters** but only **2–4** speakers | Clustering merged identities | Raise **`DIARIZATION_MAX_SPEAKERS`** (avoid `5` for cartoons); enable **`DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP`**, **`DIARIZATION_RESEGMENT_SMOOTH`**, optional **Silero VAD**. |
| Third person always **wrong label** in 3-way dialogue | Model found too few clusters | Same as above; tune **`DIARIZATION_CLUSTERING_THRESHOLD`** / **`DIARIZATION_VBX_*`** per pyannote docs. |

ASR text errors (wrong words) need a different Whisper model or language setting; this stack only improves **who spoke when** and **line grouping**.

## Troubleshooting

- **`python`: command not found** — Use `python3` or `.venv/bin/python`.
- **Diarization disabled / warnings** — Set `HF_TOKEN`; accept model conditions on Hugging Face.
- **FFmpeg / torchcodec warnings** — Install FFmpeg; on Apple Silicon, mismatched `torchcodec` vs PyTorch may warn; audio often still works via pydub/whisperx paths.
- **CUDA not used** — Install a CUDA build of PyTorch matching your driver; CPU fallback works but is slower.
- **Out of memory** — Close other apps; fewer concurrent jobs; shorter files or smaller Whisper model (would require code change from `large-v3`).

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX), [OpenAI Whisper](https://github.com/openai/whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/).
