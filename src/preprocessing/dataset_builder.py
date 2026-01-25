import json
import csv
from pathlib import Path
import logging

from src.config.settings import PROJECT_ROOT, META_DIR, GEGBE_META_DIR
# Import the aligner
from src.preprocessing.audio_alignment import align_chapter, PYDUB_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
# New folder for split audio
AUDIO_SPLIT_DIR = PROCESSED_DIR / "audio_split"
AUDIO_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR_16K = PROCESSED_DIR / "audio_16k"
OUTPUT_CSV = PROCESSED_DIR / "bible_asr_dataset.csv"

def build_asr_dataset(limit_chapters_per_lang=None):
    if not PYDUB_AVAILABLE:
        logger.error("Pydub not installed. Please install it to run alignment.")
        return

    meta_files = [
        (META_DIR / "ewe_bible_raw.json", "ewe"),
        (GEGBE_META_DIR / "gegbe_bible_raw.json", "gegbe")
    ]

    all_rows = []
    
    for meta_path, lang in meta_files:
        if not meta_path.exists():
            logger.warning(f"Meta file not found: {meta_path}")
            continue
            
        logger.info(f"Processing metadata for {lang}...")
        records = json.loads(meta_path.read_text(encoding="utf-8"))
        
        # Group by Book+Chapter
        chapters = {}
        for r in records:
            if not r.get("audio_path"):
                continue
            audio_path_key = r["audio_path"]
            if audio_path_key not in chapters:
                chapters[audio_path_key] = []
            chapters[audio_path_key].append(r)
            
        chapter_keys = list(chapters.keys())
        logger.info(f"Found {len(chapter_keys)} unique chapters for {lang}")
        
        if limit_chapters_per_lang:
            logger.info(f"Limiting to {limit_chapters_per_lang} chapters for {lang}")
            chapter_keys = chapter_keys[:limit_chapters_per_lang]

        # Process each chapter
        for count, audio_source_path in enumerate(chapter_keys):
            verses_list = chapters[audio_source_path]
            
            # Sort verses
            def verse_sort_key(v):
                try:
                    return int(str(v["verse"]).split('-')[0].strip())
                except:
                    return 0
            verses_list.sort(key=verse_sort_key)
            
            # Locate 16k Wav
            wav_name = f"{lang}_{Path(audio_source_path).with_suffix('.wav').name}"
            wav_path = AUDIO_DIR_16K / wav_name
            
            if not wav_path.exists():
                logger.debug(f"WAV 16k missing: {wav_path}")
                continue
                
            first = verses_list[0]
            book_chapter_id = f"{first['book']}_{first['chapter']}"
            
            logger.info(f"[{lang}] Alignant {book_chapter_id} ({count+1}/{len(chapter_keys)})...")
            
            # Run alignment
            aligned_rows = align_chapter(
                audio_path=wav_path,
                verses=verses_list,
                output_dir=AUDIO_SPLIT_DIR,
                lang_prefix=lang,
                book_chapter_id=book_chapter_id
            )
            
            all_rows.extend(aligned_rows)

    if not all_rows:
        logger.warning("No data rows generated.")
        return

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "language"])
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Successfully generated ASR dataset with {len(all_rows)} segments at {OUTPUT_CSV}")

if __name__ == "__main__":
    # Example: limit to 20 chapters per language to avoid overloading PC initially
    build_asr_dataset(limit_chapters_per_lang=20)

