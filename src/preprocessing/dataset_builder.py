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

def build_asr_dataset():
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
        
        # Group by Book+Chapter to reconstruct the full chapter text/audio pair
        # Structure: chapters[audio_filename] = [verse_dict, verse_dict...]
        chapters = {}
        
        for r in records:
            if not r.get("audio_path"):
                continue
            
            # Identify unique audio source key
            audio_path_key = r["audio_path"]
            if audio_path_key not in chapters:
                chapters[audio_path_key] = []
            
            chapters[audio_path_key].append(r)
            
        logger.info(f"Found {len(chapters)} unique chapters for {lang}")
        
        # Process each chapter
        for audio_source_path, verses_list in chapters.items():
            # Sort verses by verse number just in case (heuristic search for integer)
            def verse_sort_key(v):
                try:
                    return int(str(v["verse"]).split('-')[0].strip())
                except:
                    return 0
            verses_list.sort(key=verse_sort_key)
            
            # Locate 16k Wav
            # Original logic: ewe_gen_01.wav
            wav_name = f"{lang}_{Path(audio_source_path).with_suffix('.wav').name}"
            wav_path = AUDIO_DIR_16K / wav_name
            
            if not wav_path.exists():
                logger.debug(f"WAV 16k missing: {wav_path}")
                continue
                
            # Alignment ID
            # e.g. GEN_01
            first = verses_list[0]
            book_chapter_id = f"{first['book']}_{first['chapter']}"
            
            # Run alignment
            aligned_rows = align_chapter(
                audio_path=wav_path,
                verses=verses_list,
                output_dir=AUDIO_SPLIT_DIR,
                lang_prefix=lang,
                book_chapter_id=book_chapter_id
            )
            
            all_rows.extend(aligned_rows)
            
            if len(all_rows) % 100 == 0:
                logger.info(f"Processed {len(all_rows)} segments so far...")

    if not all_rows:
        logger.warning("No data rows generated.")
        return

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "language"])
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(f"Successfully generated ASR dataset with {len(all_rows)} segments at {OUTPUT_CSV}")

