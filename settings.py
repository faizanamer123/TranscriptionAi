import logging
import os
import torch
from dotenv import load_dotenv

# WhisperX checkpoints use full pickle; PyTorch 2.x defaults to weights_only=True
_original_torch_load = torch.load


def _torch_load_whisperx(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_whisperx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TranscribeAI")

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    logger.warning("HF_TOKEN not set — diarization disabled (set HF_TOKEN in .env)")
else:
    logger.info("HF_TOKEN loaded")

SAMPLE_RATE = 16000
LONG_AUDIO_SEC = 600
CHUNK_SEC = 300
CHUNK_OVERLAP_SEC = 30
JOBS_MAX = 500
