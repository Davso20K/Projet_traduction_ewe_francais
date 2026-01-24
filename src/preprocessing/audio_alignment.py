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
    
    Args:
        audio_path: Path to the full chapter WAV (16kHz).
        verses: List of dicts, each having 'text' and 'verse'.
        output_dir: Directory to save split audio files.
        lang_prefix: 'ewe' or 'gegbe'.
        book_chapter_id: e.g. 'GEN_01'.
    
    Returns:
        List of dicts: check 'dataset_builder' for expected format (audio_path, text, language).
    """
    if not PYDUB_AVAILABLE:
        logger.error("pydub not installed. Cannot align audio properly. Please install: pip install pydub")
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
    
    total_ms = len(audio)
    total_chars = sum(len(v["text"]) for v in verses)
    
    if total_chars == 0:
        return []

    # Detect Silence (expensive operation, do only if needed)
    # Min silence 500ms, threshold -40dBFS (adjust as needed)
    silence_ranges = detect_silence(audio, min_silence_len=400, silence_thresh=-40)
    # Create list of candidate cut points (mid-points of silences)
    cut_points = [(start + end) // 2 for start, end in silence_ranges]
    cut_points.sort()
    
    current_time_ms = 0
    aligned_data = []
    
    # Heuristic loop
    for i, verse in enumerate(verses):
        text = verse["text"]
        
        # Last verse gets the remainder
        if i == len(verses) - 1:
            end_time_ms = total_ms
        else:
            # Theoretical duration based on char count
            char_count = len(text)
            theoretical_duration = (char_count / total_chars) * total_ms
            proposed_end = current_time_ms + theoretical_duration
            
            # Snap to nearest silence cut point
            # Search window: +/- 3 seconds? Or just nearest?
            # Let's find nearest cut_point
            best_cut = None
            min_diff = float('inf')
            
            # Optimization: bisect could be faster but list is small
            for cp in cut_points:
                if cp <= current_time_ms: 
                    continue # specific to this implementation
                diff = abs(cp - proposed_end)
                if diff < min_diff:
                    min_diff = diff
                    best_cut = cp
                # If we passed proposed_end by a lot, stop searching
                if cp > proposed_end + 5000:
                    break
            
            # If silence is found within reasonable distance (e.g. 3s), use it
            if best_cut and min_diff < 3000:
                end_time_ms = best_cut
            else:
                end_time_ms = int(proposed_end)
        
        # Safety check
        if end_time_ms > total_ms:
            end_time_ms = total_ms
        if end_time_ms <= current_time_ms:
            # Force at least 100ms
            end_time_ms = current_time_ms + 100
            
        # Slice audio
        chunk = audio[current_time_ms:end_time_ms]
        
        # Export
        # Format: <lang>_<book>_<chapter>_<verse>.wav
        # clean verse num (can be '1' or '1-2')
        verse_clean = str(verse['verse']).replace(":", "-").replace(" ", "")
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
