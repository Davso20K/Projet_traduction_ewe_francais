import logging
import json
import math
from pathlib import Path
try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from src.config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

def align_chapter(audio_path: Path, verses: list, output_dir: Path, lang_prefix: str, book_chapter_id: str):
    """
    Aligns a long chapter audio file with its verses using a length-based heuristic
    snapped to silence.
    """
    if not PYDUB_AVAILABLE:
        logger.error("pydub not installed. Cannot align audio properly.")
        return []

    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return []

    # Load Audio
    try:
        audio = AudioSegment.from_wav(str(audio_path))
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        return []
    
    # 1. Detect active range (crop silence/intro/outro)
    # This helps if there's a long intro music or outro
    silence_ranges = detect_silence(audio, min_silence_len=500, silence_thresh=-45)
    
    # Simple heuristic to find the start and end of the actual spoken content
    start_offset = 0
    end_offset = len(audio)
    
    if silence_ranges:
        # If the first silence starts at 0, the first non-silence starts at its end
        if silence_ranges[0][0] < 500:
            start_offset = silence_ranges[0][1]
        
        # If the last silence ends at the very end, the content ends at its start
        if silence_ranges[-1][1] > len(audio) - 500:
            end_offset = silence_ranges[-1][0]

    content_audio = audio[start_offset:end_offset]
    content_ms = len(content_audio)
    total_chars = sum(len(v["text"]) for v in verses)
    
    if total_chars == 0 or content_ms == 0:
        return []

    # 2. Refined cut points within the content area
    content_silence_ranges = detect_silence(content_audio, min_silence_len=300, silence_thresh=-40)
    cut_points = [(start + end) // 2 for start, end in content_silence_ranges]
    cut_points.sort()
    
    current_time_ms = 0 # Relative to start_offset
    aligned_data = []
    
    for i, verse in enumerate(verses):
        text = verse["text"]
        
        if i == len(verses) - 1:
            end_time_ms = content_ms
        else:
            char_count = len(text)
            theoretical_duration = (char_count / total_chars) * content_ms
            proposed_end = current_time_ms + theoretical_duration
            
            # Find nearest cut_point in a window
            best_cut = None
            min_diff = float('inf')
            
            for cp in cut_points:
                if cp <= current_time_ms + 100: # avoid cutting too close to start
                    continue 
                diff = abs(cp - proposed_end)
                if diff < min_diff:
                    min_diff = diff
                    best_cut = cp
                if cp > proposed_end + 5000:
                    break
            
            # Use silence if found within 4 seconds, otherwise use heuristic
            if best_cut and min_diff < 4000:
                end_time_ms = best_cut
            else:
                end_time_ms = int(proposed_end)
        
        # Ensure minimum duration
        if end_time_ms <= current_time_ms + 200:
            end_time_ms = current_time_ms + 200
        if end_time_ms > content_ms:
            end_time_ms = content_ms
            
        # Slice from the original audio using the cumulative start_offset
        abs_start = start_offset + current_time_ms
        abs_end = start_offset + end_time_ms
        chunk = audio[abs_start:abs_end]
        
        # Clean verse num
        verse_clean = str(verse['verse']).replace(":", "-").replace(" ", "").replace("/", "-")
        file_name = f"{lang_prefix}_{book_chapter_id}_{verse_clean}.wav"
        out_path = output_dir / file_name
        
        chunk.export(out_path, format="wav")
        
        aligned_data.append({
            "audio_filepath": str(out_path),
            "text": text,
            "language": lang_prefix
        })
        
        current_time_ms = end_time_ms
        
    return aligned_data
