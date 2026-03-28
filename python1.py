import os
import sys
import shutil
import time
import asyncio
import logging
import traceback
import gc
import multiprocessing
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures

# Audio processing
import torch
import whisperx
import psutil
from pyannote.audio import Pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydub import AudioSegment
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from whisperx.diarize import DiarizationPipeline
from datetime import timedelta
from faster_whisper import WhisperModel
import torchaudio

# Fix for WhisperX weight loading
original_load = torch.load
def unsafe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = unsafe_load

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TranscribeAI")

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Validate HF_TOKEN
if not HF_TOKEN:
    logger.warning("HF_TOKEN not set - diarization will be disabled")
else:
    logger.info("HF_TOKEN loaded successfully")

def chunk_audio(audio, chunk_duration=300, sample_rate=16000, overlap=30):
    """Split audio into overlapping chunks for parallel processing"""
    import numpy as np
    
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap * sample_rate
    total_samples = len(audio)
    chunks = []
    
    # Create overlapping chunks
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio[start:end])
        
        # Move start point with overlap
        start = end - overlap_samples
        if start >= total_samples:
            break
    
    logger.info(f"Split audio into {len(chunks)} overlapping chunks of {chunk_duration}s each with {overlap}s overlap")
    return chunks

def enhance_speaker_assignment(result, confidence_threshold=0.3):
    """Log speaker assignment statistics without demoting valid assignments"""
    total_words = 0
    speaker_counts = {}
    
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        for word in words:
            total_words += 1
            speaker = word.get("speaker", "UNKNOWN")
            
            # Count speakers for analysis
            if speaker != "UNKNOWN":
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    # Calculate statistics
    detected_speakers = len(speaker_counts)
    unknown_words = sum(1 for segment in result.get("segments", []) 
                      for word in segment.get("words", []) 
                      if word.get("speaker", "UNKNOWN") == "UNKNOWN")
    unknown_ratio = unknown_words / total_words if total_words > 0 else 0
    
    # Log speaker detection info
    logger.info(f"Detected {detected_speakers} speakers: {list(speaker_counts.keys())}")
    logger.info(f"Speaker assignment: {unknown_ratio:.2%} UNKNOWN ({total_words - unknown_words}/{total_words} words assigned)")
    
    # Return result unchanged (no demotion)
    return result, (total_words - unknown_words) / total_words if total_words > 0 else 0

def detect_optimal_speakers(audio_duration, sample_rate=16000):
    """Dynamically determine optimal speaker count based on audio characteristics"""
    # Force minimum 2 speakers for interviews to prevent collapsing
    if audio_duration < 300:  # < 5 minutes
        return 2, 3  # 2-3 speakers for short audio
    elif audio_duration < 600:  # < 10 minutes
        return 2, 4  # 2-4 speakers for medium audio
    else:  # Longer conversations
        return 2, 6  # 2-6 speakers for long audio

def transcribe_chunk_only(chunk, asr_model, align_model, metadata, dev, chunk_idx):
    """Transcribe and align a chunk without diarization (diarization done globally)"""
    try:
        # Transcribe chunk
        result = asr_model.transcribe(chunk, batch_size=6)
        
        # Align chunk
        result = whisperx.align(result["segments"], align_model, metadata, chunk, dev, return_char_alignments=True)
        
        logger.info(f"Chunk {chunk_idx}: Transcription and alignment completed")
        return result
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} transcription error: {e}")
        return None

def merge_words_by_speaker(result, min_segment_duration=0.5, max_gap=0.8, allow_micro_flip=False):
    """Merge consecutive words from the same speaker into coherent segments"""
    merged_segments = []
    
    # Collect all words from all segments
    all_words = []
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        all_words.extend(words)
    
    if not all_words:
        return merged_segments
    
    # Sort words by start time
    all_words.sort(key=lambda x: x["start"])
    
    # Initialize first word
    current_speaker = all_words[0].get("speaker", "UNKNOWN")
    start_time = all_words[0]["start"]
    end_time = all_words[0]["end"]
    text_buffer = all_words[0]["word"]
    
    # Process remaining words
    for i in range(1, len(all_words)):
        word = all_words[i]
        speaker = word.get("speaker", "UNKNOWN")
        
        gap = word["start"] - end_time
        duration_so_far = end_time - start_time
        
        # Check if same speaker and small gap
        same_speaker = speaker == current_speaker
        small_gap = gap <= max_gap
        
        # Optional smoothing for tiny speaker flips (disabled by default)
        micro_flip = (
            allow_micro_flip
            and not same_speaker
            and duration_so_far < min_segment_duration
        )
        
        if (same_speaker and small_gap) or micro_flip:
            # Merge word into current segment
            text_buffer += " " + word["word"]
            end_time = word["end"]
        else:
            # Close current segment
            if (end_time - start_time) >= min_segment_duration:
                merged_segments.append({
                    "speaker": current_speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": text_buffer.strip()
                })
            
            # Start new segment
            current_speaker = speaker
            start_time = word["start"]
            end_time = word["end"]
            text_buffer = word["word"]
    
    # Add final segment
    if (end_time - start_time) >= min_segment_duration:
        merged_segments.append({
            "speaker": current_speaker,
            "start": start_time,
            "end": end_time,
            "text": text_buffer.strip()
        })
    
    return merged_segments

def clean_orphan_segments(segments, min_duration=0.8):
    """Clean orphan segments that are isolated speaker blips between same-speaker turns"""
    if not segments:
        return segments
    
    cleaned = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        prev_spk = segments[i-1]["speaker"] if i > 0 else None
        next_spk = segments[i+1]["speaker"] if i < len(segments)-1 else None
        
        # If this is a short orphan segment between same speakers, merge it
        if duration < min_duration and prev_spk == next_spk and prev_spk:
            seg = {**seg, "speaker": prev_spk}
        
        cleaned.append(seg)
    
    return cleaned

def resolve_unknown_words(result, window=3.0):
    """Resolve UNKNOWN words by finding nearest known speaker within window"""
    all_words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    known = [w for w in all_words if w.get("speaker", "UNKNOWN") != "UNKNOWN"]
    
    if not known:
        return result  # No known speakers to resolve from
    
    resolved_count = 0
    for word in all_words:
        if word.get("speaker", "UNKNOWN") == "UNKNOWN":
            mid = (word["start"] + word["end"]) / 2
            closest = min(known, key=lambda w: abs((w["start"] + w["end"]) / 2 - mid))
            if abs((closest["start"] + closest["end"]) / 2 - mid) <= window:
                word["speaker"] = closest["speaker"]
                resolved_count += 1
    
    logger.info(f"Resolved {resolved_count} UNKNOWN words using {len(known)} known speakers")
    return result

def merge_words_by_speaker(result, min_segment_duration=0.5, max_gap=1.5, allow_micro_flip=True):
    """
    Merge words by speaker to create clean conversation segments
    """
    merged_segments = []
    
    # Collect all words from all segments
    all_words = []
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        all_words.extend(words)
    
    if not all_words:
        return merged_segments
    
    # Sort words by start time
    all_words.sort(key=lambda x: x["start"])
    
    # Initialize first word
    current_speaker = all_words[0].get("speaker", "UNKNOWN")
    start_time = all_words[0]["start"]
    end_time = all_words[0]["end"]
    text_buffer = all_words[0]["word"]
    
    # Process remaining words
    for i in range(1, len(all_words)):
        word = all_words[i]
        speaker = word.get("speaker", "UNKNOWN")
        
        gap = word["start"] - end_time
        duration_so_far = end_time - start_time
        
        # Check if same speaker and small gap
        same_speaker = speaker == current_speaker
        small_gap = gap <= max_gap
        
        # Optional smoothing for tiny speaker flips
        micro_flip = (
            allow_micro_flip
            and not same_speaker
            and duration_so_far < min_segment_duration
        )
        
        if (same_speaker and small_gap) or micro_flip:
            # Merge word into current segment
            text_buffer += " " + word["word"]
            end_time = word["end"]
        else:
            # Close current segment
            if (end_time - start_time) >= min_segment_duration:
                merged_segments.append({
                    "speaker": current_speaker,
                    "start": start_time,
                    "end": end_time,
                    "text": text_buffer.strip()
                })
            
            # Start new segment
            current_speaker = speaker
            start_time = word["start"]
            end_time = word["end"]
            text_buffer = word["word"]
    
    # Add final segment
    if (end_time - start_time) >= min_segment_duration:
        merged_segments.append({
            "speaker": current_speaker,
            "start": start_time,
            "end": end_time,
            "text": text_buffer.strip()
        })
    
    return merged_segments

def format_conversation_transcript(segments, total_speakers=None):
    """
    Format merged segments into clean conversation transcript
    """
    if not segments:
        return "# Speaker Diarized Conversation\n\nNo conversation segments found."
    
    # Detect unique speakers
    unique_speakers = sorted(set(seg["speaker"] for seg in segments))
    speaker_count = len(unique_speakers) if total_speakers is None else total_speakers
    
    transcript = f"# Speaker Diarized Conversation\n\n"
    transcript += f"**Detected {speaker_count} speaker{'s' if speaker_count != 1 else ''}**\n\n"
    
    for segment in segments:
        speaker = segment["speaker"]
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        text = segment["text"]
        
        transcript += f"**{speaker} ({start_time}-{end_time})**: {text}\n\n"
    
    return transcript

def process_conversation_transcript(result, output_file=None):
    """
    Process a transcription result into a clean conversation transcript
    """
    # Resolve unknown speakers
    result = resolve_unknown_words(result)
    
    # Merge words by speaker
    merged_segments = merge_words_by_speaker(result)
    
    # Detect speaker count
    speaker_count, speakers = detect_speaker_count(result)
    
    # Format conversation transcript
    conversation_transcript = format_conversation_transcript(merged_segments, speaker_count)
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(conversation_transcript)
        print(f"Conversation transcript saved to {output_file}")
    
    return conversation_transcript, speaker_count, speakers

def detect_speaker_count(result):
    """
    Detect the number of unique speakers in the transcription
    """
    speakers = set()
    for segment in result.get("segments", []):
        words = segment.get("words", [])
        for word in words:
            if "speaker" in word:
                speakers.add(word["speaker"])
    
    return len(speakers), sorted(list(speakers))

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"

def transcribe_with_timeout(job_id, audio, asr_model, shared_state):
    """Transcription with cancel checking during the process"""
    def transcribe():
        return asr_model.transcribe(audio, batch_size=6)
    
    # Use ThreadPoolExecutor with timeout and cancel checking
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(transcribe)
        
        # Wait for completion with frequent cancel checks
        while not future.done():
            if shared_state.get(f"cancel_{job_id}"):
                logger.info(f"Job {job_id}: Cancel detected during transcription, cancelling future")
                future.cancel()
                raise KeyboardInterrupt("Transcription cancelled")
            
            # Wait a short time before checking again
            try:
                future.result(timeout=0.1)  # Check every 100ms
            except concurrent.futures.TimeoutError:
                continue
        
        return future.result()

def check_pause(job_id, shared_state):
    while shared_state.get("is_paused", False) or shared_state.get(f"pause_{job_id}", False):
        # Check for cancel during pause - check frequently
        if shared_state.get(f"cancel_{job_id}"):
            logger.info(f"Job {job_id}: Cancel detected during pause, exiting pause loop")
            return True
        time.sleep(0.1)  # Reduced from 1s to 100ms for faster cancel response
    
    # Final cancel check after pause loop
    if shared_state.get(f"cancel_{job_id}"):
        logger.info(f"Job {job_id}: Cancel detected after pause")
        return True
    return False

def clear_cancel_flag(job_id, shared_state):
    """Clear the cancel flag for a job"""
    if f"cancel_{job_id}" in shared_state:
        del shared_state[f"cancel_{job_id}"]
        logger.info(f"Job {job_id}: Cancel flag cleared")

def persistent_worker(task_queue, shared_state, job_to_worker, idx):
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if dev == "cuda" else "float32"
        
        # Each worker loads its own models - no cross-process sharing needed
        logger.info(f"Worker {idx}: Loading WhisperX model on {dev}...")
        asr_model = whisperx.load_model("large-v3", device=dev, compute_type=compute_type, language="en")
        logger.info(f"Worker {idx}: WhisperX model loaded successfully")
        
        logger.info(f"Worker {idx}: Loading alignment model...")
        align_model, metadata = whisperx.load_align_model("en", device=dev)
        logger.info(f"Worker {idx}: Alignment model loaded successfully")
        
        diarize_model = None
        # Load diarization model
        try:
            logger.info(f"Worker {idx}: Attempting to load diarization model...")
            diarize_model = DiarizationPipeline(token=HF_TOKEN, device=dev)
            logger.info(f"Worker {idx}: Diarization model loaded successfully")
        except Exception as e:
            logger.error(f"Worker {idx}: Failed to load diarization model: {e}")
            diarize_model = None
        
        logger.info(f"Worker {idx}: Initialization complete, ready for jobs")
        
        while True:
            job_id, path = None, None
            task_started = False
            try:
                job_id, path = task_queue.get()
                if job_id is None: break
                task_started = True  # Mark that we've taken the task from queue
                
                # Check if job was cancelled before worker picked it up
                if shared_state.get(f"cancel_{job_id}"):
                    logger.info(f"Worker {idx}: Job {job_id} was cancelled before processing, skipping")
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                    clear_cancel_flag(job_id, shared_state)
                    task_queue.task_done() if task_started else None
                    continue
                    
                if check_pause(job_id, shared_state): 
                    # If check_pause returns True, it means either cancelled or still paused
                    if shared_state.get(f"cancel_{job_id}"):
                        # Job was cancelled, update status and exit
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        continue
                    else:
                        # Still paused, continue to next iteration
                        continue

                file_ext = os.path.splitext(path)[1].replace(".", "").lower()
                sound = AudioSegment.from_file(path, format=file_ext)
                sound = sound.set_channels(1).set_frame_rate(16000) # mono and 16khz sample rate
                
                # Audio preprocessing for better accuracy
                from pydub.effects import normalize
                sound = normalize(sound)  # Normalize audio levels
                sound = sound.high_pass_filter(60)  # Remove low-frequency rumble (safe for male voices)
                
                clean_path = path + "_clean.wav"; sound.export(clean_path, format="wav")
                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "transcribing", "progress": 15}
                if check_pause(job_id, shared_state): 
                    # If check_pause returns True, it means either cancelled or still paused
                    if shared_state.get(f"cancel_{job_id}"):
                        # Job was cancelled, update status and exit
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    else:
                        # Still paused, continue to next iteration
                        continue

                audio = whisperx.load_audio(clean_path)
                logger.info(f"Audio loaded successfully for job {job_id}, shape: {audio.shape if hasattr(audio, 'shape') else 'No shape'}")
                
                # Check if audio is long enough for chunking
                audio_duration = len(audio) / 16000  # 16kHz sample rate
                if audio_duration > 600:  # 10 minutes threshold
                    logger.info(f"Job {job_id}: Long audio detected ({audio_duration:.1f}s), using chunked parallel processing")
                    
                    # Split into chunks for parallel processing
                    chunks = chunk_audio(audio, chunk_duration=300)  # 5-minute chunks
                    chunk_results = []
                    
                    # Process chunks in parallel with limited workers to avoid over-parallelization
                    # Limit to 2 workers to avoid resource overload with multiprocessing
                    max_chunk_workers = min(len(chunks), 2)
                    logger.info(f"Job {job_id}: Processing {len(chunks)} chunks with {max_chunk_workers} workers")
                    
                    # Check for cancel before starting chunk processing
                    if shared_state.get(f"cancel_{job_id}"):
                        logger.info(f"Job {job_id}: Cancelled before chunk processing")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    
                    cancelled_during_chunks = False
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_chunk_workers) as executor:
                        future_to_chunk = {
                            executor.submit(transcribe_chunk_only, chunk, asr_model, align_model, metadata, dev, i): i
                            for i, chunk in enumerate(chunks)
                        }
                        
                        # Collect results in order
                        chunk_results = [None] * len(chunks)
                        for future in concurrent.futures.as_completed(future_to_chunk):
                            # Check for cancel during chunk processing
                            if shared_state.get(f"cancel_{job_id}"):
                                logger.info(f"Job {job_id}: Cancelled during chunk processing")
                                # Cancel remaining futures
                                for f in future_to_chunk:
                                    f.cancel()
                                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                                clear_cancel_flag(job_id, shared_state)
                                cancelled_during_chunks = True
                                break
                                
                            chunk_idx = future_to_chunk[future]
                            try:
                                result = future.result()
                                if result:
                                    chunk_results[chunk_idx] = result
                                    logger.info(f"Job {job_id}: Chunk {chunk_idx} transcription completed")
                            except Exception as e:
                                logger.error(f"Job {job_id}: Chunk {chunk_idx} failed: {e}")
                    
                    # Check if we cancelled during chunk processing
                    if cancelled_during_chunks:
                        task_queue.task_done() if task_started else None
                        continue
                    
                    # Check for cancel after chunk processing
                    if shared_state.get(f"cancel_{job_id}"):
                        logger.info(f"Job {job_id}: Cancelled after chunk processing")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    
                    # Merge chunk results with correct ordering and proper timestamp handling
                    if chunk_results:
                        # Check for cancel before merging
                        if shared_state.get(f"cancel_{job_id}"):
                            logger.info(f"Job {job_id}: Cancelled before merging")
                            shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                            clear_cancel_flag(job_id, shared_state)
                            task_queue.task_done() if task_started else None
                            continue
                            
                        # Combine all segments from all chunks with proper timestamp handling
                        all_segments = []
                        time_offset = 0
                        
                        for chunk_idx, chunk_result in enumerate(chunk_results):
                            if chunk_result is None:
                                continue  # Skip failed chunks
                                
                            for segment in chunk_result.get("segments", []):
                                # Create a copy to avoid modifying original
                                segment_copy = segment.copy()
                                # Adjust timestamps by adding the current time offset
                                segment_copy["start"] += time_offset
                                segment_copy["end"] += time_offset
                                
                                # Fix word-level timestamps too
                                if "words" in segment_copy:
                                    for word in segment_copy["words"]:
                                        word["start"] += time_offset
                                        word["end"] += time_offset
                                
                                all_segments.append(segment_copy)
                            
                            # Update time offset for next chunk (account for overlap)
                            chunk_duration = len(chunks[chunk_idx]) / 16000 if chunk_idx < len(chunks) else 0
                            if chunk_idx < len(chunks) - 1:  # Not last chunk
                                time_offset += chunk_duration - 30  # Subtract overlap
                            else:
                                time_offset += chunk_duration
                        
                        # Create merged result for diarization
                        merged_result = {"segments": all_segments}
                        
                        # Remove duplicate words from overlap zones
                        logger.info(f"Job {job_id}: Deduplicating {len(all_segments)} segments...")
                        seen_starts = set()
                        deduped_segments = []
                        
                        for segment in all_segments:
                            if "words" in segment:
                                # Deduplicate words within segment
                                deduped_words = []
                                word_starts = set()
                                
                                for word in segment["words"]:
                                    word_start = round(word["start"], 3)  # Round to handle floating point precision
                                    if word_start not in word_starts:
                                        word_starts.add(word_start)
                                        deduped_words.append(word)
                                
                                segment["words"] = deduped_words
                            
                            # Deduplicate segments by start time
                            seg_start = round(segment["start"], 3)
                            if seg_start not in seen_starts:
                                seen_starts.add(seg_start)
                                deduped_segments.append(segment)
                        
                        merged_result["segments"] = deduped_segments
                        logger.info(f"Job {job_id}: Deduplicated to {len(deduped_segments)} segments")
                        
                        # Global word deduplication across all segments
                        logger.info(f"Job {job_id}: Performing cross-segment word deduplication...")
                        global_word_starts = set()
                        total_words_before = sum(len(seg.get("words", [])) for seg in deduped_segments)
                        
                        for segment in deduped_segments:
                            clean_words = []
                            for word in segment.get("words", []):
                                key = round(word["start"], 2)
                                if key not in global_word_starts:
                                    global_word_starts.add(key)
                                    clean_words.append(word)
                            segment["words"] = clean_words
                        
                        total_words_after = sum(len(seg.get("words", [])) for seg in deduped_segments)
                        logger.info(f"Job {job_id}: Cross-segment deduplication: {total_words_before} → {total_words_after} words")
                        
                        logger.info(f"Job {job_id}: Merged {len([r for r in chunk_results if r])} chunks into {len(all_segments)} total segments")
                        
                        # Now diarize the FULL audio for global context
                        if diarize_model and not shared_state.get(f"cancel_{job_id}"):
                            # Check for cancel before global diarization
                            if shared_state.get(f"cancel_{job_id}"):
                                logger.info(f"Job {job_id}: Cancelled before global diarization")
                                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                                clear_cancel_flag(job_id, shared_state)
                                task_queue.task_done() if task_started else None
                                continue
                                
                            try:
                                logger.info(f"Job {job_id}: Running global diarization on full audio...")
                                min_speakers, max_speakers = detect_optimal_speakers(audio_duration)
                                logger.info(f"Job {job_id}: Using adaptive speaker range {min_speakers}-{max_speakers}")
                                
                                # Diarize the FULL audio for consistent speaker labels
                                global_diarization = diarize_model(
                                    audio,
                                    min_speakers=min_speakers,
                                    max_speakers=max_speakers
                                )
                                
                                # Check if job was cancelled during global diarization
                                if shared_state.get(f"cancel_{job_id}"):
                                    logger.info(f"Job {job_id}: Cancelled during global diarization")
                                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                                    clear_cancel_flag(job_id, shared_state)
                                    task_queue.task_done() if task_started else None
                                    continue
                                
                                # Assign speakers from global diarization to merged segments
                                result = whisperx.assign_word_speakers(global_diarization, merged_result)
                                
                                # Log speaker assignment accuracy
                                result, accuracy = enhance_speaker_assignment(result)
                                logger.info(f"Job {job_id}: Speaker assignment accuracy: {accuracy:.2%}")
                                
                                logger.info(f"Job {job_id}: Global diarization completed and applied")
                            except Exception as de:
                                logger.error(f"Job {job_id}: Global diarization error: {de}")
                                result = merged_result  # Fallback to merged result
                        else:
                            result = merged_result
                    else:
                        logger.error(f"Job {job_id}: All chunks failed")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "failed", "error": "All chunks failed"}
                        continue
                else:
                    # Use original sequential processing for short audio
                    logger.info(f"Job {job_id}: Short audio ({audio_duration:.1f}s), using sequential processing")
                    
                    logger.info(f"Starting transcription for job {job_id}...")
                    # Check for cancel before transcription
                    if shared_state.get(f"cancel_{job_id}"):
                        logger.info(f"Job {job_id}: Cancelled before transcription")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                        
                    try:
                        result = transcribe_with_timeout(job_id, audio, asr_model, shared_state)
                        logger.info(f"Transcription completed for job {job_id}")
                    except KeyboardInterrupt:
                        logger.info(f"Job {job_id}: Transcription cancelled by user")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    except Exception as e:
                        logger.error(f"Transcription error for job {job_id}: {e}")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "failed", "error": str(e)}
                        continue
                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "aligning", "progress": 40}
                if check_pause(job_id, shared_state): 
                    # If check_pause returns True, it means either cancelled or still paused
                    if shared_state.get(f"cancel_{job_id}"):
                        # Job was cancelled, update status and exit
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    else:
                        # Still paused, continue to next iteration
                        continue

                # Only align for short audio path (long audio already aligned in chunks)
                if audio_duration <= 600:
                    logger.info(f"Starting alignment for job {job_id}...")
                    result = whisperx.align(result["segments"], align_model, metadata, audio, dev, return_char_alignments=True)
                    logger.info(f"Alignment completed for job {job_id}")
                    gc.collect(); torch.cuda.empty_cache()
                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "diarizing", "progress": 60}
                if check_pause(job_id, shared_state): 
                    # If check_pause returns True, it means either cancelled or still paused
                    if shared_state.get(f"cancel_{job_id}"):
                        # Job was cancelled, update status and exit
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    else:
                        # Still paused, continue to next iteration
                        continue

                # Only diarize for short audio (long audio already diarized globally)
                if audio_duration <= 600 and diarize_model:
                    # Check for cancel before starting diarization
                    if shared_state.get(f"cancel_{job_id}"):
                        logger.info(f"Job {job_id}: Cancelled before diarization")
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                        
                    try:
                        logger.info(f"Running diarization for job {job_id}...")
                        # Use adaptive speaker detection for consistency
                        min_speakers, max_speakers = detect_optimal_speakers(audio_duration)
                        logger.info(f"Job {job_id}: Using adaptive speaker range {min_speakers}-{max_speakers}")
                        
                        diarization = diarize_model(
                            audio,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers
                        )
                        
                        # Check if job was cancelled during diarization
                        if shared_state.get(f"cancel_{job_id}"):
                            logger.info(f"Job {job_id}: Cancelled during diarization")
                            shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                            clear_cancel_flag(job_id, shared_state)
                            task_queue.task_done() if task_started else None
                            continue
                        
                        result = whisperx.assign_word_speakers(diarization, result)
                        
                        # Log speaker assignment accuracy
                        result, accuracy = enhance_speaker_assignment(result)
                        logger.info(f"Job {job_id}: Speaker assignment accuracy: {accuracy:.2%}")
                        
                        # Printing word-level diarization output as requested
                        for segment in result.get("segments", []):
                            words = segment.get("words", [])
                            for word in words:
                                speaker = word.get("speaker", "UNKNOWN")
                                start = word["start"]
                                end = word["end"]
                                text = word["word"]
                                print(f"{speaker} ({start:.2f}-{end:.2f}): {text}")
                                
                    except Exception as de:
                        logger.error(f"Diarization error for job {job_id}: {de}")
                        import traceback
                        logger.error(f"Diarization traceback: {traceback.format_exc()}")
                
                shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "finalizing", "progress": 90}
                if check_pause(job_id, shared_state): 
                    # If check_pause returns True, it means either cancelled or still paused
                    if shared_state.get(f"cancel_{job_id}"):
                        # Job was cancelled, update status and exit
                        shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                        clear_cancel_flag(job_id, shared_state)
                        task_queue.task_done() if task_started else None
                        continue
                    else:
                        # Still paused, continue to next iteration
                        continue

                # Create clean conversation transcript
                result = resolve_unknown_words(result)
                merged_segments = merge_words_by_speaker(result)
                speaker_count, speakers = detect_speaker_count(result)
                conversation_transcript = format_conversation_transcript(merged_segments, speaker_count)
                
                if shared_state.get(f"cancel_{job_id}"):
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "cancelled", "progress": 0}
                    clear_cancel_flag(job_id, shared_state)  # Clear cancel flag
                else:
                    os.makedirs("transcriptions", exist_ok=True)
                    md_filename = f"{job_id}_{os.path.basename(path)}.md"
                    md_path = os.path.join("transcriptions", md_filename)
                    
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(conversation_transcript)
                    
                    logger.info(f"Job {job_id}: Conversation transcript saved to {md_path}")
                    logger.info(f"Job {job_id}: Detected {speaker_count} speakers: {', '.join(speakers)}")
                    
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "completed", "progress": 100, "md_file": md_filename}
                    clear_cancel_flag(job_id, shared_state)  # Clear cancel flag
                
                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                if job_id is not None: 
                    logger.error(f"Worker {os.getpid()} task error on job {job_id}: {e}")
                    logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Update job status with detailed error info
                    shared_state[job_id] = {**shared_state.get(job_id, {}), "status": "failed", "error": f"{type(e).__name__}: {str(e)}"}
            finally:
                if job_id is not None: 
                    task_queue.task_done() if task_started else None
                    
                    # Clean up temporary files
                    if 'clean_path' in locals() and os.path.exists(clean_path): 
                        os.remove(clean_path)
                    
                    # Memory cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    except Exception as e: logger.error(f"Worker {os.getpid()} initialization error: {e}")

def detect_hardware():
    """Detect system hardware capabilities for optimal worker configuration"""
    logical_cores = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 4
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate optimal worker count based on hardware
    if gpu_available:
        # GPU systems can handle more workers
        base_workers = min(physical_cores * 2, gpu_count * 3)
    else:
        # CPU-only systems
        base_workers = max(2, physical_cores // 2)
    
    # Memory limit (leave 2GB for system)
    memory_limit_gb = max(1, int((total_ram_gb - 2) // 1.2))
    
    # Battery detection
    battery = psutil.sensors_battery()
    on_battery = battery.power_plugged is False if battery else False
    
    return (
        logical_cores,      # Total logical cores
        physical_cores,      # Physical cores for performance
        gpu_available,       # GPU availability
        gpu_count,          # Number of GPUs
        base_workers,       # Calculated optimal worker count
        total_ram_gb,      # Total system RAM
        memory_limit_gb,    # Available memory for workers
        on_battery          # Battery status
    )

L_CORES, P_CORES, GPU_ON, GPU_COUNT, BASE_WORKERS, RAM_GB, MEM_LIM, ON_BATT = detect_hardware()
manager, shared_state, task_queue, job_to_worker, worker_pool, lock = None, None, None, None, [], None
jobs, active_ids, cpu_hist, last_scale = [], set(), [], 0

def sync_pool(target):
    global worker_pool
    worker_pool = [p for p in worker_pool if p.is_alive()]
    curr = len(worker_pool)
    if curr < target:
        for i in range(curr, target):
            p = multiprocessing.Process(target=persistent_worker, args=(task_queue, shared_state, job_to_worker, i), daemon=True)
            p.start(); worker_pool.append(p)
    elif curr > target:
        for _ in range(curr - target): task_queue.put((None, None))

async def dispatcher():
    while True:
        try:
            # Always update job statuses and active_ids even when paused
            with lock:  # Add lock for thread safety
                for j in jobs:
                    if j["id"] in shared_state:
                        s = shared_state[j["id"]]
                        if j["status"] in ["cancelled", "completed", "failed"] and s.get("status") in ["processing", "paused"]:
                            continue
                        j.update(s)
                        if j["status"] in ["completed", "failed", "cancelled"] and os.path.exists(j["path"]):
                            try: os.remove(j["path"])
                            except: pass
                
                # Enforce dynamic limits: Auto-pause extra jobs if above current capacity
                allowed = shared_state.get("allowed_workers", BASE_WORKERS)
                # Exclude cancelled jobs from active jobs list
                active_jobs = [j for j in jobs if j["status"] not in ["completed", "failed", "cancelled", "waiting"]]
                active_jobs.sort(key=lambda x: x["id"]) # Sort by ID to prioritize older jobs
                
                for i, j in enumerate(active_jobs):
                    if i < allowed:
                        if shared_state.get(f"pause_{j['id']}") and not shared_state.get("is_paused"):
                            shared_state[f"pause_{j['id']}"] = False
                    else:
                        shared_state[f"pause_{j['id']}"] = True
                
                active_ids.clear(); active_ids.update([j["id"] for j in active_jobs if not shared_state.get(f"pause_{j['id']}")])
                
                # Only dispatch new jobs if NOT paused and under limit
                if not shared_state.get("is_paused", False):
                    for j in [j for j in jobs if j["status"] == "waiting"]:
                        # Double-check job wasn't cancelled while waiting
                        if shared_state.get(f"cancel_{j['id']}"):
                            logger.info(f"Dispatcher: Job {j['id']} was cancelled, skipping dispatch")
                            j["status"] = "cancelled"
                            shared_state[j["id"]] = {**shared_state.get(j["id"], {}), "status": "cancelled", "progress": 0}
                            clear_cancel_flag(j["id"], shared_state)
                            continue
                            
                        if len(active_ids) >= allowed: break
                        j["status"] = "transcribing"; active_ids.add(j["id"])
                        shared_state[j["id"]] = {"status": "transcribing", "progress": 2}
                        task_queue.put((j["id"], j["path"]))
        except (EOFError, ConnectionResetError, BrokenPipeError): break
        except Exception as e: logger.error(f"Dispatcher error: {e}")
        await asyncio.sleep(0.5)

async def cpu_monitor():
    global last_scale
    while True:
        try:
            batt = psutil.sensors_battery()
            on_b = batt.power_plugged is False if batt else False
            shared_state["is_on_battery"] = on_b
            cpu, mem, q_size = psutil.cpu_percent(), psutil.virtual_memory().percent, len([j for j in jobs if j["status"] == "waiting"])
            cpu_hist.append(cpu)
            if len(cpu_hist) > 10: cpu_hist.pop(0)
            avg_cpu, curr, now = sum(cpu_hist)/len(cpu_hist), shared_state["allowed_workers"], asyncio.get_event_loop().time()
            lim = curr
            if cpu > 95 or mem > 92: lim, last_scale = max(1, curr-1), now
            elif (now - last_scale) > 10:
                if on_b:
                    b_cap = max(1, BASE_WORKERS // 2)
                    lim = b_cap if curr > b_cap else (min(curr+1, b_cap) if avg_cpu < 20 else curr)
                else:
                    if curr < BASE_WORKERS and avg_cpu < 65: lim = curr + 1
                    elif (q_size > 0 or avg_cpu < 30) and mem < 75: lim = min(curr+1, P_CORES, MEM_LIM)
                    elif avg_cpu > 80: lim = max(BASE_WORKERS, curr-1)
            if lim != curr:
                shared_state["allowed_workers"] = lim; sync_pool(lim); last_scale = now
        except (EOFError, ConnectionResetError, BrokenPipeError): break
        except Exception as e: logger.error(f"CPU Monitor error: {e}")
        await asyncio.sleep(4)

@asynccontextmanager
async def lifespan(app):
    global manager, shared_state, task_queue, job_to_worker, lock
    manager = multiprocessing.Manager(); shared_state, task_queue, job_to_worker, lock = manager.dict(), manager.JoinableQueue(), manager.dict(), manager.Lock()
    shared_state.update({"allowed_workers": BASE_WORKERS, "is_paused": False, "job_id_counter": 0, "is_on_battery": ON_BATT})
    sync_pool(BASE_WORKERS); asyncio.create_task(dispatcher()); asyncio.create_task(cpu_monitor())
    yield
    for p in worker_pool:
        if p.is_alive(): p.terminate()
    manager.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def ui(): return FileResponse("index.html")

@app.get("/download/{filename}")
async def download(filename: str):
    # Security: Sanitize filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    
    # Security: Validate filename format
    if not safe_filename or safe_filename != filename:
        raise HTTPException(400, "Invalid filename")
    
    # Security: Only allow .md files
    if not safe_filename.endswith('.md'):
        raise HTTPException(400, "Only markdown files are allowed")
    
    path = os.path.join("transcriptions", safe_filename)
    
    # Security: Ensure path is within transcriptions directory
    if not os.path.abspath(path).startswith(os.path.abspath("transcriptions")):
        raise HTTPException(400, "Access denied")
    
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    
    return FileResponse(path, media_type='text/markdown', filename=safe_filename)

@app.post("/upload")
async def upload(files: list[UploadFile]):
    os.makedirs("uploads", exist_ok=True); added = []
    with lock:
        for f in files:
            if not f.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.mkv')): continue
            jid = shared_state["job_id_counter"]
            # Ensure the counter exists and is updated atomically
            shared_state["job_id_counter"] = jid + 1
            path = os.path.join("uploads", f"{jid}_{f.filename}")
            with open(path, "wb") as buf: shutil.copyfileobj(f.file, buf)
            j = {"id": jid, "file": f.filename, "path": path, "status": "waiting", "progress": 0, "result": "", "error": ""}
            jobs.append(j); added.append(j)
    if not added: raise HTTPException(400, "No valid files")
    return {"status": "uploaded", "jobs": added}

@app.get("/status")
async def status():
    active_jobs_count = len([j for j in jobs if j["status"] not in ["completed", "failed", "cancelled", "waiting", "paused"]])
    return {"cpu_usage": psutil.cpu_percent(), "max_workers": shared_state["allowed_workers"], "base_workers": BASE_WORKERS, "active_workers": active_jobs_count, "queue_size": len([j for j in jobs if j["status"] == "waiting"]), "is_paused": shared_state["is_paused"], "is_on_battery": shared_state["is_on_battery"], "total_ram": f"{RAM_GB:.1f} GB", "jobs": jobs[::-1][:50]}

@app.post("/pause")
async def pause():
    shared_state["is_paused"] = True
    for j in jobs:
        if j["status"] not in ["completed", "failed", "cancelled", "waiting", "paused"]:
            j["prev_status"] = j["status"] # Store current state
            j["status"] = "paused"
            shared_state[j["id"]] = {**shared_state.get(j["id"], {}), "status": "paused"}
    return {"status": "paused"}

@app.post("/resume")
async def resume():
    shared_state["is_paused"] = False
    for j in jobs:
        if j["status"] == "paused":
            restored_status = j.get("prev_status", "transcribing") # Default to transcribing
            j["status"] = restored_status
            shared_state[j["id"]] = {**shared_state.get(j["id"], {}), "status": restored_status}
    return {"status": "resumed"}

@app.post("/cancel/{jid}")
async def cancel(jid: int):
    j = next((j for j in jobs if j["id"] == jid), None)
    if not j: raise HTTPException(404)
    
    logger.info(f"Cancelling job {jid} with status: {j.get('status', 'unknown')}")
    
    if j["status"] in ["waiting", "processing", "transcribing", "aligning", "diarizing", "finalizing"]:
        # Set cancel flag
        shared_state[f"cancel_{jid}"] = True
        
        # Update job status immediately
        j["status"] = "cancelled"
        j["progress"] = 0
        shared_state[jid] = {**shared_state.get(jid, {}), "status": "cancelled", "progress": 0}
        
        # Remove from active_ids if it's there
        if jid in active_ids:
            active_ids.remove(jid)
            logger.info(f"Removed job {jid} from active_ids")
        
        # If job is waiting, remove it from queue immediately
        if j["status"] == "waiting":
            logger.info(f"Job {jid} was waiting, cancelled successfully")
            # No need to remove from queue since we set status to cancelled
            # and dispatcher checks status before dispatching
                
    logger.info(f"Job {jid} cancel flag set and status updated to cancelled")
    return {"status": "cancelled", "job_id": jid}

@app.delete("/clear")
async def clear():
    for p in worker_pool:
        try: p.kill()
        except: pass
    sync_pool(shared_state["allowed_workers"]); jobs.clear(); active_ids.clear(); return {"status": "cleared"}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config="uvicorn.config")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down gracefully...")
        # Cancel all active jobs before shutdown
        for job in jobs:
            if job["status"] not in ["completed", "failed", "cancelled"]:
                shared_state[f"cancel_{job['id']}"] = True
                job["status"] = "cancelled"
        
        # Give workers time to finish current tasks
        time.sleep(2)
        
        # Terminate workers gracefully
        for worker in worker_pool:
            try:
                worker.terminate(timeout=5)
            except:
                try:
                    worker.kill()
                except:
                    pass
        
        logger.info("All workers terminated")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        logger.info("Application shutdown complete")
