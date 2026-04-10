import re
from collections import Counter

from settings import (
    TRANSCRIPT_CLEAN_ORPHANS,
    TRANSCRIPT_COLLAPSE_ISLANDS,
    TRANSCRIPT_ISLAND_MAX_SEC,
    TRANSCRIPT_MERGE_ALLOW_MICRO_FLIP,
    TRANSCRIPT_MERGE_MAX_GAP_SEC,
    TRANSCRIPT_MERGE_MICRO_FLIP_SEC,
    TRANSCRIPT_MAX_SEGMENT_SEC,
    TRANSCRIPT_MIN_SPEAKER_TURN_SEC,
    TRANSCRIPT_ORPHAN_MIN_DURATION_SEC,
    TRANSCRIPT_RESOLVE_MAX_PASSES,
    TRANSCRIPT_RESOLVE_UNKNOWN,
    TRANSCRIPT_RESOLVE_WINDOW_SEC,
    TRANSCRIPT_SMOOTH_PASSES,
    TRANSCRIPT_SMOOTH_RADIUS,
    TRANSCRIPT_SMOOTH_SPEAKERS,
    TRANSCRIPT_TEMPORAL_GLUE,
    TRANSCRIPT_TEMPORAL_GLUE_LONG_CAP_SEC,
    TRANSCRIPT_TEMPORAL_GLUE_MAX_GAP_SEC,
    TRANSCRIPT_TEMPORAL_GLUE_SHORT_SEC,
    TRANSCRIPT_TEMPORAL_GLUE_ULTRA_GAP_SEC,
    logger,
)


def _iter_words_chronological(result):
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w:
                words.append(w)
    words.sort(key=lambda x: x["start"])
    return words


def enhance_speaker_assignment(result):
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
    min_segment_duration=None,
    max_gap=None,
    allow_micro_flip=None,
):
    if min_segment_duration is None:
        min_segment_duration = TRANSCRIPT_MERGE_MICRO_FLIP_SEC
    if max_gap is None:
        max_gap = TRANSCRIPT_MERGE_MAX_GAP_SEC
    if allow_micro_flip is None:
        allow_micro_flip = TRANSCRIPT_MERGE_ALLOW_MICRO_FLIP
    max_segment_sec = TRANSCRIPT_MAX_SEGMENT_SEC

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
            # Prevent long speaker runs from swallowing neighboring turns when labels drift.
            if same_speaker and duration_so_far >= max_segment_sec:
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
                continue
            text_buffer += " " + word["word"]
            end_time = word["end"]
        else:
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

    merged_segments.append(
        {
            "speaker": current_speaker,
            "start": start_time,
            "end": end_time,
            "text": text_buffer.strip(),
        }
    )

    return merged_segments


def glue_temporal_fragments(
    segments,
    max_gap_sec=None,
    ultra_gap_sec=None,
    short_sec=None,
    long_cap_sec=None,
):
    """Join adjacent lines when timing looks like diarization jitter, not a real turn change."""
    if max_gap_sec is None:
        max_gap_sec = TRANSCRIPT_TEMPORAL_GLUE_MAX_GAP_SEC
    if ultra_gap_sec is None:
        ultra_gap_sec = TRANSCRIPT_TEMPORAL_GLUE_ULTRA_GAP_SEC
    if short_sec is None:
        short_sec = TRANSCRIPT_TEMPORAL_GLUE_SHORT_SEC
    if long_cap_sec is None:
        long_cap_sec = TRANSCRIPT_TEMPORAL_GLUE_LONG_CAP_SEC

    if len(segments) < 2:
        return segments

    out = []
    cur = dict(segments[0])
    for j in range(1, len(segments)):
        nxt = dict(segments[j])
        gap = float(nxt["start"]) - float(cur["end"])
        d0 = max(float(cur["end"]) - float(cur["start"]), 1e-6)
        d1 = max(float(nxt["end"]) - float(nxt["start"]), 1e-6)

        if gap <= ultra_gap_sec and cur["speaker"] == nxt["speaker"]:
            merge = True
        elif gap <= max_gap_sec:
            small_side = min(d0, d1) <= short_sec
            # Only merge likely boundary jitter: one side short and overall not too long
            not_both_long_turns = max(d0, d1) <= long_cap_sec
            merge = cur["speaker"] == nxt["speaker"] and small_side and not_both_long_turns
        else:
            merge = False

        if merge:
            votes = Counter({cur["speaker"]: d0, nxt["speaker"]: d1})
            spk = votes.most_common(1)[0][0]
            cur = {
                "speaker": spk,
                "start": cur["start"],
                "end": max(float(cur["end"]), float(nxt["end"])),
                "text": re.sub(r"\s+", " ", f"{cur['text']} {nxt['text']}").strip(),
            }
        else:
            out.append(cur)
            cur = nxt
    out.append(cur)
    return out


def clean_orphan_segments(segments, min_duration=None):
    if min_duration is None:
        min_duration = TRANSCRIPT_ORPHAN_MIN_DURATION_SEC
    if not segments:
        return segments
    cleaned = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        prev_spk = segments[i - 1]["speaker"] if i > 0 else None
        next_spk = segments[i + 1]["speaker"] if i < len(segments) - 1 else None
        sp = seg.get("speaker")
        if sp == "UNKNOWN" and prev_spk and prev_spk == next_spk:
            seg = {**seg, "speaker": prev_spk}
        elif duration < min_duration and prev_spk == next_spk and prev_spk:
            seg = {**seg, "speaker": prev_spk}
        cleaned.append(seg)
    return cleaned


def stabilize_short_turns(segments, min_turn_sec=None):
    """
    Anti-false-speaker rule:
    do not keep isolated very short turns as separate speakers unless evidence is strong.
    """
    if min_turn_sec is None:
        min_turn_sec = TRANSCRIPT_MIN_SPEAKER_TURN_SEC
    if len(segments) < 3:
        return segments

    out = [dict(s) for s in segments]
    speaker_total = Counter()
    for s in out:
        speaker_total[s["speaker"]] += max(0.0, float(s["end"]) - float(s["start"]))

    for i in range(1, len(out) - 1):
        prev_seg, cur_seg, next_seg = out[i - 1], out[i], out[i + 1]
        cur_dur = max(0.0, float(cur_seg["end"]) - float(cur_seg["start"]))
        if cur_dur >= min_turn_sec:
            continue
        if prev_seg["speaker"] == next_seg["speaker"] and cur_seg["speaker"] != prev_seg["speaker"]:
            # Strong evidence: same speaker on both sides with a short middle blip.
            cur_seg["speaker"] = prev_seg["speaker"]
            continue

        # If this speaker has very little presence overall, attach to stronger adjacent context.
        if speaker_total[cur_seg["speaker"]] < (min_turn_sec * 2):
            prev_dur = max(0.0, float(prev_seg["end"]) - float(prev_seg["start"]))
            next_dur = max(0.0, float(next_seg["end"]) - float(next_seg["start"]))
            cur_seg["speaker"] = prev_seg["speaker"] if prev_dur >= next_dur else next_seg["speaker"]

    merged = []
    cur = dict(out[0])
    for nxt in out[1:]:
        if nxt["speaker"] == cur["speaker"] and float(nxt["start"]) - float(cur["end"]) <= 0.35:
            cur["end"] = max(float(cur["end"]), float(nxt["end"]))
            cur["text"] = re.sub(r"\s+", " ", f"{cur['text']} {nxt['text']}").strip()
        else:
            merged.append(cur)
            cur = dict(nxt)
    merged.append(cur)
    return merged


def resolve_unknown_words(result, window_sec=None, max_passes=None):
    if window_sec is None:
        window_sec = TRANSCRIPT_RESOLVE_WINDOW_SEC
    if max_passes is None:
        max_passes = TRANSCRIPT_RESOLVE_MAX_PASSES

    all_words = _iter_words_chronological(result)
    if not all_words:
        return result

    total_fixed = 0
    for _ in range(max_passes):
        changed = 0
        known_centers = [
            ((w["start"] + w["end"]) / 2, w.get("speaker", "UNKNOWN"))
            for w in all_words
            if w.get("speaker", "UNKNOWN") != "UNKNOWN"
        ]
        for word in all_words:
            if word.get("speaker", "UNKNOWN") != "UNKNOWN":
                continue
            mid = (word["start"] + word["end"]) / 2
            votes = [
                sp
                for t, sp in known_centers
                if sp != "UNKNOWN" and abs(t - mid) <= window_sec
            ]
            if votes:
                word["speaker"] = Counter(votes).most_common(1)[0][0]
                changed += 1
                continue
            labeled = [w for w in all_words if w.get("speaker", "UNKNOWN") != "UNKNOWN"]
            if labeled:
                nearest = min(
                    labeled,
                    key=lambda w: abs((w["start"] + w["end"]) / 2 - mid),
                )
                word["speaker"] = nearest["speaker"]
                changed += 1
        total_fixed += changed
        if changed == 0:
            break

    if total_fixed:
        logger.info(
            f"Resolved UNKNOWN tokens ({total_fixed} label updates, window={window_sec}s)"
        )
    return result


def smooth_word_level_speakers(
    result, window_radius=None, passes=None,
):
    if window_radius is None:
        window_radius = TRANSCRIPT_SMOOTH_RADIUS
    if passes is None:
        passes = TRANSCRIPT_SMOOTH_PASSES

    words = _iter_words_chronological(result)
    if len(words) < 3:
        return result

    labels = [w.get("speaker", "UNKNOWN") for w in words]
    for _ in range(passes):
        new_labels = []
        for i in range(len(words)):
            lo = max(0, i - window_radius)
            hi = min(len(words), i + window_radius + 1)
            neigh = [labels[j] for j in range(lo, hi) if labels[j] != "UNKNOWN"]
            if len(neigh) >= 3:
                spk, cnt = Counter(neigh).most_common(1)[0]
                ratio = cnt / max(1, len(neigh))
                cur = labels[i]
                # Conservative smoothing: only relabel with strong local majority
                # or when current label is UNKNOWN.
                if cur == "UNKNOWN" or ratio >= 0.8:
                    new_labels.append(spk)
                else:
                    new_labels.append(cur)
            elif len(neigh) == 1:
                new_labels.append(neigh[0])
            else:
                new_labels.append(labels[i])
        labels = new_labels
        for w, sp in zip(words, labels):
            w["speaker"] = sp
    return result


def collapse_short_speaker_islands(result, max_island_sec=None):
    if max_island_sec is None:
        max_island_sec = TRANSCRIPT_ISLAND_MAX_SEC

    words = _iter_words_chronological(result)
    if len(words) < 3:
        return result

    i = 0
    while i < len(words):
        j = i
        sp = words[i].get("speaker", "UNKNOWN")
        while j < len(words) and words[j].get("speaker", "UNKNOWN") == sp:
            j += 1
        dur = words[j - 1]["end"] - words[i]["start"]
        left = words[i - 1].get("speaker") if i > 0 else None
        right = words[j].get("speaker") if j < len(words) else None
        if (
            left
            and right
            and left == right
            and sp != left
            and sp != "UNKNOWN"
            and dur <= max_island_sec
        ):
            for k in range(i, j):
                words[k]["speaker"] = left
        elif (
            left
            and right
            and left == right
            and sp == "UNKNOWN"
            and dur <= max_island_sec * 2
        ):
            for k in range(i, j):
                words[k]["speaker"] = left
        i = j
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


def build_conversation_markdown(result):
    if TRANSCRIPT_RESOLVE_UNKNOWN:
        result = resolve_unknown_words(result)
    if TRANSCRIPT_SMOOTH_SPEAKERS:
        result = smooth_word_level_speakers(result)
    if TRANSCRIPT_COLLAPSE_ISLANDS:
        result = collapse_short_speaker_islands(result)
    merged = merge_words_by_speaker(result)
    merged = stabilize_short_turns(merged)
    if TRANSCRIPT_TEMPORAL_GLUE:
        merged = glue_temporal_fragments(merged)
    if TRANSCRIPT_CLEAN_ORPHANS:
        merged = clean_orphan_segments(merged)

    labels = sorted({seg["speaker"] for seg in merged})
    non_unknown = [s for s in labels if s != "UNKNOWN"]
    if non_unknown:
        speaker_count, speakers = len(non_unknown), non_unknown
    else:
        speaker_count, speakers = len(labels), labels
    text = format_conversation_transcript(merged, speaker_count)
    return text, speaker_count, speakers
