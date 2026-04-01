from settings import logger

def enhance_speaker_assignment(result, confidence_threshold=0.3):
    _ = confidence_threshold
    total_words = 0
    speaker_counts = {}

    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            total_words += 1
            speaker = word.get("speaker", "UNKNOWN")
            if speaker != "UNKNOWN":
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    unknown_words = sum(
        1
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
        if word.get("speaker", "UNKNOWN") == "UNKNOWN"
    )
    unknown_ratio = unknown_words / total_words if total_words > 0 else 0

    logger.info(f"Speakers detected: {list(speaker_counts.keys())}")
    logger.info(
        f"Word-level assignment: {1 - unknown_ratio:.2%} labeled ({total_words - unknown_words}/{total_words})"
    )

    accuracy = (total_words - unknown_words) / total_words if total_words > 0 else 0
    return result, accuracy


def merge_words_by_speaker(
    result,
    min_segment_duration=0.35,
    max_gap=1.15,
    allow_micro_flip=True,
):
    # Turn word-level labels into readable blocks; micro_flip smooths 1-word label glitches
    merged_segments = []
    all_words = []
    for segment in result.get("segments", []):
        all_words.extend(segment.get("words", []))

    if not all_words:
        return merged_segments

    all_words.sort(key=lambda x: x["start"])

    current_speaker = all_words[0].get("speaker", "UNKNOWN")
    start_time = all_words[0]["start"]
    end_time = all_words[0]["end"]
    text_buffer = all_words[0]["word"]

    for i in range(1, len(all_words)):
        word = all_words[i]
        speaker = word.get("speaker", "UNKNOWN")
        gap = word["start"] - end_time
        duration_so_far = end_time - start_time
        same_speaker = speaker == current_speaker
        small_gap = gap <= max_gap
        micro_flip = (
            allow_micro_flip
            and not same_speaker
            and duration_so_far < min_segment_duration
        )

        if (same_speaker and small_gap) or micro_flip:
            text_buffer += " " + word["word"]
            end_time = word["end"]
        else:
            if (end_time - start_time) >= min_segment_duration:
                merged_segments.append(
                    {
                        "speaker": current_speaker,
                        "start": start_time,
                        "end": end_time,
                        "text": text_buffer.strip(),
                    }
                )
            current_speaker = speaker
            start_time = word["start"]
            end_time = word["end"]
            text_buffer = word["word"]

    if (end_time - start_time) >= min_segment_duration:
        merged_segments.append(
            {
                "speaker": current_speaker,
                "start": start_time,
                "end": end_time,
                "text": text_buffer.strip(),
            }
        )

    return merged_segments


def clean_orphan_segments(segments, min_duration=0.55):
    # Short single-speaker blips between the same speaker on both sides → relabel
    if not segments:
        return segments
    cleaned = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        prev_spk = segments[i - 1]["speaker"] if i > 0 else None
        next_spk = segments[i + 1]["speaker"] if i < len(segments) - 1 else None
        if duration < min_duration and prev_spk == next_spk and prev_spk:
            seg = {**seg, "speaker": prev_spk}
        cleaned.append(seg)
    return cleaned


def resolve_unknown_words(result, window_sec=2.5):
    # Map UNKNOWN words to nearest timed labeled word within window (helps at boundaries)
    all_words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    known = [w for w in all_words if w.get("speaker", "UNKNOWN") != "UNKNOWN"]
    if not known:
        return result

    resolved_count = 0
    for word in all_words:
        if word.get("speaker", "UNKNOWN") != "UNKNOWN":
            continue
        mid = (word["start"] + word["end"]) / 2
        closest = min(known, key=lambda w: abs((w["start"] + w["end"]) / 2 - mid))
        if abs((closest["start"] + closest["end"]) / 2 - mid) <= window_sec:
            word["speaker"] = closest["speaker"]
            resolved_count += 1

    logger.info(f"Resolved {resolved_count} UNKNOWN tokens (window={window_sec}s)")
    return result


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def format_conversation_transcript(segments, total_speakers=None):
    if not segments:
        return "# Speaker Diarized Conversation\n\nNo conversation segments found."

    unique_speakers = sorted({seg["speaker"] for seg in segments})
    speaker_count = len(unique_speakers) if total_speakers is None else total_speakers

    lines = [
        "# Speaker Diarized Conversation\n",
        f"**Detected {speaker_count} speaker{'s' if speaker_count != 1 else ''}**\n\n",
    ]
    for segment in segments:
        speaker = segment["speaker"]
        t0 = format_time(segment["start"])
        t1 = format_time(segment["end"])
        text = segment["text"]
        lines.append(f"**{speaker} ({t0}-{t1})**: {text}\n\n")
    return "".join(lines)


def detect_speaker_count(result):
    speakers = set()
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            if "speaker" in word:
                speakers.add(word["speaker"])
    return len(speakers), sorted(speakers)


def build_conversation_markdown(result):
    # Post-diarization pipeline: fill UNKNOWN → merge lines → strip micro-orphans → markdown
    result = resolve_unknown_words(result)
    merged = merge_words_by_speaker(result)
    merged = clean_orphan_segments(merged)
    speaker_count, speakers = detect_speaker_count(result)
    text = format_conversation_transcript(merged, speaker_count)
    return text, speaker_count, speakers
