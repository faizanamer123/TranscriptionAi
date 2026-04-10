from settings import logger

def chunk_audio(audio, chunk_duration=300, sample_rate=16000, overlap=30):
    # Overlapping chunks keep context at boundaries; overlap trimmed later when merging
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap * sample_rate
    total_samples = len(audio)
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio[start:end])
        start = end - overlap_samples
        if start >= total_samples:
            break
    logger.info(
        f"Split audio into {len(chunks)} chunks ({chunk_duration}s, {overlap}s overlap)"
    )
    return chunks
