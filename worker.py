import concurrent.futures
import gc
import os
import time
import traceback
import torch
import whisperx
from pydub import AudioSegment
from chunking import chunk_audio
from diarization import diarize_to_dataframe, load_pyannote_pipeline
from settings import (
    CHUNK_OVERLAP_SEC,
    CHUNK_SEC,
    DIARIZATION_FALLBACK_MODEL,
    DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MODEL,
    HF_TOKEN,
    LONG_AUDIO_SEC,
    SAMPLE_RATE,
    TRANSCRIPT_ASSIGN_FILL_NEAREST,
    logger,
)
from transcript import build_conversation_markdown, enhance_speaker_assignment


def _diarize_kwargs():
    kw = {}
    if DIARIZATION_MIN_SPEAKERS is not None:
        kw["min_speakers"] = DIARIZATION_MIN_SPEAKERS
    if DIARIZATION_MAX_SPEAKERS is not None:
        kw["max_speakers"] = DIARIZATION_MAX_SPEAKERS
    return kw


def transcribe_chunk_only(chunk, asr_model, align_model, metadata, dev, chunk_idx):
    try:
        result = asr_model.transcribe(chunk, batch_size=6)
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            chunk,
            dev,
            return_char_alignments=True,
        )
        logger.info(f"Chunk {chunk_idx}: transcribe + align done")
        return result
    except Exception as e:
        logger.error(f"Chunk {chunk_idx}: {e}")
        return None


def transcribe_with_timeout(job_id, audio, asr_model, shared_state):
    def run():
        return asr_model.transcribe(audio, batch_size=6)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run)
        while not future.done():
            if shared_state.get(f"cancel_{job_id}"):
                future.cancel()
                raise KeyboardInterrupt("cancelled")
            try:
                future.result(timeout=0.1)
            except concurrent.futures.TimeoutError:
                continue
        return future.result()


def check_pause(job_id, shared_state):
    # Blocks while system or job is paused; returns True if cancelled
    while shared_state.get("is_paused", False) or shared_state.get(
        f"pause_{job_id}", False
    ):
        if shared_state.get(f"cancel_{job_id}"):
            logger.info(f"Job {job_id}: cancel during pause")
            return True
        time.sleep(0.1)
    if shared_state.get(f"cancel_{job_id}"):
        logger.info(f"Job {job_id}: cancel after pause")
        return True
    return False


def clear_cancel_flag(job_id, shared_state):
    if f"cancel_{job_id}" in shared_state:
        del shared_state[f"cancel_{job_id}"]
        logger.info(f"Job {job_id}: cancel flag cleared")


def persistent_worker(task_queue, shared_state, _unused_job_map, idx):
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if dev == "cuda" else "float32"

        logger.info(f"Worker {idx}: loading WhisperX on {dev}…")
        asr_model = whisperx.load_model(
            "large-v3", device=dev, compute_type=compute_type, language="en"
        )
        align_model, metadata = whisperx.load_align_model("en", device=dev)

        diarize_model = None
        if HF_TOKEN:
            torch_dev = torch.device(dev)
            for model_name in (DIARIZATION_MODEL, DIARIZATION_FALLBACK_MODEL):
                if not model_name:
                    continue
                diarize_model = load_pyannote_pipeline(
                    model_name, HF_TOKEN, torch_dev, cache_dir=None
                )
                if diarize_model is not None:
                    logger.info(f"Worker {idx}: diarization ({model_name}) ready")
                    break
            if diarize_model is None:
                logger.error(f"Worker {idx}: no diarization model could be loaded")
        else:
            logger.warning(f"Worker {idx}: no HF_TOKEN — skipping diarization")

        while True:
            job_id, path = None, None
            task_started = False
            try:
                job_id, path = task_queue.get()
                if job_id is None:
                    task_queue.task_done()
                    break
                task_started = True

                if shared_state.get(f"cancel_{job_id}"):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                if check_pause(job_id, shared_state):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                file_ext = os.path.splitext(path)[1].replace(".", "").lower()
                sound = AudioSegment.from_file(path, format=file_ext)
                sound = sound.set_channels(1).set_frame_rate(SAMPLE_RATE)

                from pydub.effects import normalize

                sound = normalize(sound)
                sound = sound.high_pass_filter(60)

                clean_path = path + "_clean.wav"
                sound.export(clean_path, format="wav")
                shared_state[job_id] = {
                    **shared_state.get(job_id, {}),
                    "status": "transcribing",
                    "progress": 15,
                }

                if check_pause(job_id, shared_state):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                audio = whisperx.load_audio(clean_path)
                audio_duration = len(audio) / SAMPLE_RATE

                if audio_duration > LONG_AUDIO_SEC:
                    logger.info(
                        f"Job {job_id}: long file ({audio_duration:.0f}s) — chunked ASR + global diarize"
                    )
                    chunks = chunk_audio(
                        audio,
                        chunk_duration=CHUNK_SEC,
                        sample_rate=SAMPLE_RATE,
                        overlap=int(CHUNK_OVERLAP_SEC),
                    )
                    max_chunk_workers = min(len(chunks), 2)
                    if shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue

                    chunk_results = [None] * len(chunks)
                    cancelled = False
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_chunk_workers
                    ) as executor:
                        future_to_chunk = {
                            executor.submit(
                                transcribe_chunk_only,
                                chunk,
                                asr_model,
                                align_model,
                                metadata,
                                dev,
                                i,
                            ): i
                            for i, chunk in enumerate(chunks)
                        }
                        for future in concurrent.futures.as_completed(future_to_chunk):
                            if shared_state.get(f"cancel_{job_id}"):
                                for f in future_to_chunk:
                                    f.cancel()
                                shared_state[job_id] = {
                                    **shared_state.get(job_id, {}),
                                    "status": "cancelled",
                                    "progress": 0,
                                }
                                clear_cancel_flag(job_id, shared_state)
                                cancelled = True
                                break
                            chunk_idx = future_to_chunk[future]
                            try:
                                r = future.result()
                                if r:
                                    chunk_results[chunk_idx] = r
                            except Exception as e:
                                logger.error(f"Job {job_id} chunk {chunk_idx}: {e}")

                    if cancelled:
                        continue

                    if shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue

                    if not any(chunk_results):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "failed",
                            "error": "All chunks failed",
                        }
                        continue

                    if shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue

                    # Map each chunks local timestamps to global time. Advance offset even if a
                    # chunk failed -> otherwise later chunks drift (overlap step must stay in sync).
                    all_segments = []
                    time_offset = 0.0
                    overlap_sec = float(CHUNK_OVERLAP_SEC)
                    n_chunks = len(chunks)
                    for chunk_idx in range(n_chunks):
                        chunk_result = (
                            chunk_results[chunk_idx]
                            if chunk_idx < len(chunk_results)
                            else None
                        )
                        if chunk_result is not None:
                            for segment in chunk_result.get("segments", []):
                                seg = segment.copy()
                                seg["start"] += time_offset
                                seg["end"] += time_offset
                                for w in seg.get("words", []):
                                    w["start"] += time_offset
                                    w["end"] += time_offset
                                all_segments.append(seg)
                        chunk_len_sec = len(chunks[chunk_idx]) / SAMPLE_RATE
                        if chunk_idx < n_chunks - 1:
                            time_offset += max(0.0, chunk_len_sec - overlap_sec)

                    merged_result = {"segments": all_segments}
                    seen_starts = set()
                    deduped = []
                    for segment in all_segments:
                        if "words" in segment:
                            ws, seen_w = [], set()
                            for w in segment["words"]:
                                k = round(w["start"], 3)
                                if k not in seen_w:
                                    seen_w.add(k)
                                    ws.append(w)
                            segment["words"] = ws
                        sk = round(segment["start"], 3)
                        if sk not in seen_starts:
                            seen_starts.add(sk)
                            deduped.append(segment)
                    merged_result["segments"] = deduped

                    global_word_starts = set()
                    for segment in deduped:
                        cw = []
                        for w in segment.get("words", []):
                            k = round(w["start"], 2)
                            if k not in global_word_starts:
                                global_word_starts.add(k)
                                cw.append(w)
                        segment["words"] = cw

                    if diarize_model and not shared_state.get(f"cancel_{job_id}"):
                        if shared_state.get(f"cancel_{job_id}"):
                            shared_state[job_id] = {
                                **shared_state.get(job_id, {}),
                                "status": "cancelled",
                                "progress": 0,
                            }
                            clear_cancel_flag(job_id, shared_state)
                            continue
                        try:
                            global_diarization = diarize_to_dataframe(
                                diarize_model, audio, **_diarize_kwargs()
                            )
                            if shared_state.get(f"cancel_{job_id}"):
                                shared_state[job_id] = {
                                    **shared_state.get(job_id, {}),
                                    "status": "cancelled",
                                    "progress": 0,
                                }
                                clear_cancel_flag(job_id, shared_state)
                                continue
                            result = whisperx.assign_word_speakers(
                                global_diarization,
                                merged_result,
                                fill_nearest=TRANSCRIPT_ASSIGN_FILL_NEAREST,
                            )
                            result, acc = enhance_speaker_assignment(result)
                            logger.info(f"Job {job_id}: assignment coverage {acc:.2%}")
                        except Exception as de:
                            logger.error(f"Job {job_id}: global diarize: {de}")
                            result = merged_result
                    else:
                        result = merged_result
                else:
                    logger.info(f"Job {job_id}: short file ({audio_duration:.0f}s) — single pass")
                    if shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue
                    try:
                        result = transcribe_with_timeout(
                            job_id, audio, asr_model, shared_state
                        )
                    except KeyboardInterrupt:
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue
                    except Exception as e:
                        logger.error(f"Job {job_id} transcribe: {e}")
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "failed",
                            "error": str(e),
                        }
                        continue

                shared_state[job_id] = {
                    **shared_state.get(job_id, {}),
                    "status": "aligning",
                    "progress": 40,
                }
                if check_pause(job_id, shared_state):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                if audio_duration <= LONG_AUDIO_SEC:
                    result = whisperx.align(
                        result["segments"],
                        align_model,
                        metadata,
                        audio,
                        dev,
                        return_char_alignments=True,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                shared_state[job_id] = {
                    **shared_state.get(job_id, {}),
                    "status": "diarizing",
                    "progress": 60,
                }
                if check_pause(job_id, shared_state):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                if audio_duration <= LONG_AUDIO_SEC and diarize_model:
                    if shared_state.get(f"cancel_{job_id}"):
                        shared_state[job_id] = {
                            **shared_state.get(job_id, {}),
                            "status": "cancelled",
                            "progress": 0,
                        }
                        clear_cancel_flag(job_id, shared_state)
                        continue
                    try:
                        diarization = diarize_to_dataframe(
                            diarize_model, audio, **_diarize_kwargs()
                        )
                        if shared_state.get(f"cancel_{job_id}"):
                            shared_state[job_id] = {
                                **shared_state.get(job_id, {}),
                                "status": "cancelled",
                                "progress": 0,
                            }
                            clear_cancel_flag(job_id, shared_state)
                            continue
                        result = whisperx.assign_word_speakers(
                            diarization,
                            result,
                            fill_nearest=TRANSCRIPT_ASSIGN_FILL_NEAREST,
                        )
                        result, acc = enhance_speaker_assignment(result)
                        logger.info(f"Job {job_id}: assignment coverage {acc:.2%}")
                    except Exception as de:
                        logger.error(f"Job {job_id} diarize: {de}")
                        logger.error(traceback.format_exc())

                shared_state[job_id] = {
                    **shared_state.get(job_id, {}),
                    "status": "finalizing",
                    "progress": 90,
                }
                if check_pause(job_id, shared_state):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                    continue

                (
                    conversation_transcript,
                    speaker_count,
                    speakers,
                ) = build_conversation_markdown(result)

                if shared_state.get(f"cancel_{job_id}"):
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "cancelled",
                        "progress": 0,
                    }
                    clear_cancel_flag(job_id, shared_state)
                else:
                    os.makedirs("transcriptions", exist_ok=True)
                    md_filename = f"{job_id}_{os.path.basename(path)}.md"
                    md_path = os.path.join("transcriptions", md_filename)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(conversation_transcript)
                    logger.info(f"Job {job_id}: saved {md_path} ({speaker_count} speakers)")
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "completed",
                        "progress": 100,
                        "md_file": md_filename,
                        "result": conversation_transcript[:8000],
                    }

                    clear_cancel_flag(job_id, shared_state)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                if job_id is not None:
                    logger.error(f"Worker job {job_id}: {e}\n{traceback.format_exc()}")
                    shared_state[job_id] = {
                        **shared_state.get(job_id, {}),
                        "status": "failed",
                        "error": f"{type(e).__name__}: {e}",
                    }
            finally:
                if job_id is not None:
                    if task_started:
                        task_queue.task_done()
                    cp = path + "_clean.wav" if path else None
                    if cp and os.path.exists(cp):
                        try:
                            os.remove(cp)
                        except OSError:
                            pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Worker {idx} init error: {e}")
