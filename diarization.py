# Wraps pyannote SpeakerDiarization from Hugging Face. Optional: Silero VAD mask, pipeline
# hyperparameters from env, and optional post-smoothing on the diarization table (all opt-in).

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation

from settings import (
    DIARIZATION_CLUSTERING_THRESHOLD,
    DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP,
    DIARIZATION_MIN_DURATION_OFF,
    DIARIZATION_RESEGMENT_ISLAND_SEC,
    DIARIZATION_RESEGMENT_MERGE_GAP_SEC,
    DIARIZATION_RESEGMENT_SMOOTH,
    DIARIZATION_SEGMENTATION_THRESHOLD,
    DIARIZATION_SILERO_PAD_MS,
    DIARIZATION_SILERO_VAD,
    DIARIZATION_USE_EXCLUSIVE,
    DIARIZATION_VBX_FA,
    DIARIZATION_VBX_FB,
    SAMPLE_RATE,
    logger,
)

_silero_state: dict[str, Any] = {"model": None, "get_ts": None, "ok": None}


def load_pyannote_pipeline(
    model_name: str,
    token: str | None,
    device: torch.device,
    cache_dir: str | None = None,
) -> Pipeline | None:
    try:
        p = Pipeline.from_pretrained(model_name, token=token, cache_dir=cache_dir)
        p.to(device)
        apply_pipeline_tuning(p)
        if DIARIZATION_EMBEDDING_EXCLUDE_OVERLAP and hasattr(
            p, "embedding_exclude_overlap"
        ):
            p.embedding_exclude_overlap = True
            logger.info("Diarization: embedding_exclude_overlap=True")
        return p
    except Exception as e:
        logger.warning(f"Failed to load pyannote pipeline {model_name}: {e}")
        return None


def apply_pipeline_tuning(pipeline: Pipeline) -> None:
    # Segmentation binarization (reduces false speech in silence)
    if hasattr(pipeline, "segmentation"):
        seg = pipeline.segmentation
        if DIARIZATION_SEGMENTATION_THRESHOLD is not None and hasattr(
            seg, "threshold"
        ):
            seg.threshold = float(DIARIZATION_SEGMENTATION_THRESHOLD)
            logger.info(
                f"Diarization: segmentation.threshold={DIARIZATION_SEGMENTATION_THRESHOLD}"
            )
        if DIARIZATION_MIN_DURATION_OFF is not None and hasattr(
            seg, "min_duration_off"
        ):
            seg.min_duration_off = float(DIARIZATION_MIN_DURATION_OFF)
            logger.info(
                f"Diarization: segmentation.min_duration_off={DIARIZATION_MIN_DURATION_OFF}"
            )

    if not hasattr(pipeline, "clustering"):
        return
    cl = pipeline.clustering
    if DIARIZATION_CLUSTERING_THRESHOLD is not None and hasattr(cl, "threshold"):
        cl.threshold = float(DIARIZATION_CLUSTERING_THRESHOLD)
        logger.info(
            f"Diarization: clustering.threshold={DIARIZATION_CLUSTERING_THRESHOLD}"
        )
    if DIARIZATION_VBX_FA is not None and hasattr(cl, "Fa"):
        cl.Fa = float(DIARIZATION_VBX_FA)
        logger.info(f"Diarization: clustering.Fa={DIARIZATION_VBX_FA}")
    if DIARIZATION_VBX_FB is not None and hasattr(cl, "Fb"):
        cl.Fb = float(DIARIZATION_VBX_FB)
        logger.info(f"Diarization: clustering.Fb={DIARIZATION_VBX_FB}")


def _ensure_silero():
    if _silero_state["ok"] is False:
        return False
    if _silero_state["model"] is not None:
        return True
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, _, _, _) = utils
        _silero_state["model"] = model
        _silero_state["get_ts"] = get_speech_timestamps
        _silero_state["ok"] = True
        logger.info("Silero VAD loaded (torch.hub)")
        return True
    except Exception as e:
        logger.warning(f"Silero VAD unavailable, skipping mask: {e}")
        _silero_state["ok"] = False
        return False


def maybe_apply_silero_vad(audio: np.ndarray) -> np.ndarray:
    if not DIARIZATION_SILERO_VAD:
        return audio
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32, copy=False)
    if not _ensure_silero():
        return audio

    wav = torch.from_numpy(audio)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    get_speech_timestamps = _silero_state["get_ts"]
    model = _silero_state["model"]

    try:
        stamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.5,
            min_speech_duration_ms=200,
            min_silence_duration_ms=100,
            speech_pad_ms=int(DIARIZATION_SILERO_PAD_MS),
            return_seconds=True,
        )
    except Exception as e:
        logger.warning(f"Silero get_speech_timestamps failed: {e}")
        return audio

    if not stamps:
        logger.info("Silero VAD: no speech detected; leaving audio unchanged for diarization")
        return audio

    n = audio.shape[0]
    mask = np.zeros(n, dtype=np.float32)
    for t in stamps:
        a = max(0, int(float(t["start"]) * SAMPLE_RATE))
        b = min(n, int(float(t["end"]) * SAMPLE_RATE))
        if b > a:
            mask[a:b] = 1.0

    out = audio * mask
    return out.astype(np.float32, copy=False)


def diarize_to_dataframe(
    pipeline: Pipeline,
    audio: np.ndarray,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> pd.DataFrame:
    x = maybe_apply_silero_vad(audio)
    file = {
        "uri": "utterance",
        "waveform": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)[None, :]),
        "sample_rate": SAMPLE_RATE,
    }
    out = pipeline(
        file,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    if (
        DIARIZATION_USE_EXCLUSIVE
        and hasattr(out, "exclusive_speaker_diarization")
        and out.exclusive_speaker_diarization is not None
    ):
        ann: Annotation = out.exclusive_speaker_diarization
    else:
        ann: Annotation = (
            out.speaker_diarization if hasattr(out, "speaker_diarization") else out
        )
    df = pd.DataFrame(
        ann.itertracks(yield_label=True),
        columns=["segment", "label", "speaker"],
    )
    df["start"] = df["segment"].apply(lambda s: s.start)
    df["end"] = df["segment"].apply(lambda s: s.end)
    df = df.drop(columns=["segment", "label"])
    if DIARIZATION_RESEGMENT_SMOOTH:
        df = resegment_smooth_df(df)
    return df


def merge_adjacent_same_speaker(df: pd.DataFrame, max_gap_sec: float) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    cur = {"start": df.at[0, "start"], "end": df.at[0, "end"], "speaker": df.at[0, "speaker"]}
    for i in range(1, len(df)):
        r = df.iloc[i]
        sp, st, en = r["speaker"], float(r["start"]), float(r["end"])
        if sp == cur["speaker"] and st - cur["end"] <= max_gap_sec:
            cur["end"] = max(cur["end"], en)
        else:
            rows.append(cur)
            cur = {"start": st, "end": en, "speaker": sp}
    rows.append(cur)
    return pd.DataFrame(rows)


def collapse_short_islands_df(df: pd.DataFrame, max_island_sec: float) -> pd.DataFrame:
    if len(df) < 3:
        return df
    rows = df.sort_values("start").to_dict("records")
    for i in range(1, len(rows) - 1):
        st, en, sp = rows[i]["start"], rows[i]["end"], rows[i]["speaker"]
        if float(en) - float(st) > max_island_sec:
            continue
        if (
            rows[i - 1]["speaker"] == rows[i + 1]["speaker"]
            and sp != rows[i - 1]["speaker"]
        ):
            rows[i]["speaker"] = rows[i - 1]["speaker"]
    out = pd.DataFrame(rows)
    return merge_adjacent_same_speaker(out, max_gap_sec=1e-3)


def resegment_smooth_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    gap = float(DIARIZATION_RESEGMENT_MERGE_GAP_SEC)
    island = float(DIARIZATION_RESEGMENT_ISLAND_SEC)
    df = merge_adjacent_same_speaker(df, max_gap_sec=gap)
    df = collapse_short_islands_df(df, max_island_sec=island)
    df = merge_adjacent_same_speaker(df, max_gap_sec=gap)
    return df.reset_index(drop=True)
