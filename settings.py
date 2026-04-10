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


def _optional_int_env(name: str, default: str | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("", "none", "inf", "unlimited"):
        return None
    v = int(s)
    if v < 1:
        return None
    return v


# pyannote/speaker-diarization-3.1 bundles segmentation-3.0 + newer clustering
DIARIZATION_MODEL = os.environ.get(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
)
DIARIZATION_FALLBACK_MODEL = os.environ.get(
    "DIARIZATION_FALLBACK_MODEL",
    "pyannote/speaker-diarization-community-1",
)
# Let the model pick count between min and max (omit max via env = no upper cap)
DIARIZATION_MIN_SPEAKERS = _optional_int_env("DIARIZATION_MIN_SPEAKERS", "1")
DIARIZATION_MAX_SPEAKERS = _optional_int_env("DIARIZATION_MAX_SPEAKERS", "20")


def _truthy_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _optional_float_env(name: str) -> float | None:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return None
    return float(v)


def _float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


# Optional Silero VAD mask before pyannote (reduces silence false alarms; needs first-time hub download)
DIARIZATION_SILERO_VAD = _truthy_env("DIARIZATION_SILERO_VAD", True)
DIARIZATION_SILERO_PAD_MS = _int_env("DIARIZATION_SILERO_PAD_MS", 40)
# Post-hoc boundary smoothing on pyannote segments (helps overlapy / multi-voice audio)
DIARIZATION_RESEGMENT_SMOOTH = _truthy_env("DIARIZATION_RESEGMENT_SMOOTH", True)
DIARIZATION_RESEGMENT_MERGE_GAP_SEC = _float_env(
    "DIARIZATION_RESEGMENT_MERGE_GAP_SEC", 0.08
)
DIARIZATION_RESEGMENT_ISLAND_SEC = _float_env(
    "DIARIZATION_RESEGMENT_ISLAND_SEC", 0.3
)
# Prefer non-overlap frames for ECAPA-style embeddings when supported
DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP = _truthy_env(
    "DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP", True
)
# Override pyannote instantiated hyperparameters (cosine AHC / VBx — see pyannote.audio docs)
DIARIZATION_SEGMENTATION_THRESHOLD = _optional_float_env(
    "DIARIZATION_SEGMENTATION_THRESHOLD"
)
DIARIZATION_MIN_DURATION_OFF = _optional_float_env("DIARIZATION_MIN_DURATION_OFF")
DIARIZATION_CLUSTERING_THRESHOLD = _optional_float_env(
    "DIARIZATION_CLUSTERING_THRESHOLD"
)
DIARIZATION_VBX_FA = _optional_float_env("DIARIZATION_VBX_FA")
DIARIZATION_VBX_FB = _optional_float_env("DIARIZATION_VBX_FB")
DIARIZATION_USE_EXCLUSIVE = _truthy_env("DIARIZATION_USE_EXCLUSIVE", True)

# Transcript: time-based merge + optional refinements (defaults tuned for mixed dialogue / animation)
TRANSCRIPT_MERGE_MAX_GAP_SEC = _float_env("TRANSCRIPT_MERGE_MAX_GAP_SEC", 0.3)
TRANSCRIPT_MERGE_MICRO_FLIP_SEC = _float_env("TRANSCRIPT_MERGE_MICRO_FLIP_SEC", 0.32)
TRANSCRIPT_MAX_SEGMENT_SEC = _float_env("TRANSCRIPT_MAX_SEGMENT_SEC", 3.5)
TRANSCRIPT_MIN_SPEAKER_TURN_SEC = _float_env("TRANSCRIPT_MIN_SPEAKER_TURN_SEC", 1.2)
TRANSCRIPT_MERGE_ALLOW_MICRO_FLIP = _truthy_env(
    "TRANSCRIPT_MERGE_ALLOW_MICRO_FLIP", False
)
# Fill UNKNOWN from nearby labeled words (temporal only)
TRANSCRIPT_RESOLVE_UNKNOWN = _truthy_env("TRANSCRIPT_RESOLVE_UNKNOWN", True)
TRANSCRIPT_RESOLVE_WINDOW_SEC = _float_env("TRANSCRIPT_RESOLVE_WINDOW_SEC", 4.5)
TRANSCRIPT_RESOLVE_MAX_PASSES = _int_env("TRANSCRIPT_RESOLVE_MAX_PASSES", 5)
TRANSCRIPT_ASSIGN_FILL_NEAREST = _truthy_env("TRANSCRIPT_ASSIGN_FILL_NEAREST", False)
TRANSCRIPT_SMOOTH_SPEAKERS = _truthy_env("TRANSCRIPT_SMOOTH_SPEAKERS", True)
TRANSCRIPT_SMOOTH_RADIUS = _int_env("TRANSCRIPT_SMOOTH_RADIUS", 1)
TRANSCRIPT_SMOOTH_PASSES = _int_env("TRANSCRIPT_SMOOTH_PASSES", 1)
TRANSCRIPT_COLLAPSE_ISLANDS = _truthy_env("TRANSCRIPT_COLLAPSE_ISLANDS", True)
TRANSCRIPT_ISLAND_MAX_SEC = _float_env("TRANSCRIPT_ISLAND_MAX_SEC", 0.45)
TRANSCRIPT_CLEAN_ORPHANS = _truthy_env("TRANSCRIPT_CLEAN_ORPHANS", False)
TRANSCRIPT_ORPHAN_MIN_DURATION_SEC = _float_env(
    "TRANSCRIPT_ORPHAN_MIN_DURATION_SEC", 0.5
)
# Merge markdown lines when gap + duration pattern looks like diarization jitter (no text rules)
TRANSCRIPT_TEMPORAL_GLUE = _truthy_env("TRANSCRIPT_TEMPORAL_GLUE", True)
TRANSCRIPT_TEMPORAL_GLUE_MAX_GAP_SEC = _float_env(
    "TRANSCRIPT_TEMPORAL_GLUE_MAX_GAP_SEC", 0.25
)
TRANSCRIPT_TEMPORAL_GLUE_ULTRA_GAP_SEC = _float_env(
    "TRANSCRIPT_TEMPORAL_GLUE_ULTRA_GAP_SEC", 0.08
)
TRANSCRIPT_TEMPORAL_GLUE_SHORT_SEC = _float_env(
    "TRANSCRIPT_TEMPORAL_GLUE_SHORT_SEC", 0.8
)
TRANSCRIPT_TEMPORAL_GLUE_LONG_CAP_SEC = _float_env(
    "TRANSCRIPT_TEMPORAL_GLUE_LONG_CAP_SEC", 2.2
)

SAMPLE_RATE = 16000
LONG_AUDIO_SEC = 600
CHUNK_SEC = 300
CHUNK_OVERLAP_SEC = 30
JOBS_MAX = 500
