import json
import csv
import logging
from pathlib import Path
from src.config.settings import META_DIR, GEGBE_META_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_parallel_dataset():
    ewe_meta = META_DIR / "ewe_bible_raw.json"
    gegbe_meta = GEGBE_META_DIR / "gegbe_bible_raw.json"
    output_csv = PROCESSED_DIR / "mina_ewe_parallel.csv"

    if not ewe_meta.exists() or not gegbe_meta.exists():
        logger.error("Metadata files for Ewe or Gegbe not found.")
        return

    logger.info("Loading Ewe and Gegbe metadata...")
    ewe_data = json.loads(ewe_meta.read_text(encoding="utf-8"))
    gegbe_data = json.loads(gegbe_meta.read_text(encoding="utf-8"))

    # Index by book_chapter_verse
    def build_index(data):
        index = {}
        for r in data:
            key = f"{r['book']}_{r['chapter']}_{r['verse']}"
            index[key] = r['text']
        return index

    ewe_index = build_index(ewe_data)
    gegbe_index = build_index(gegbe_data)

    parallel_pairs = []
    
    logger.info("Matching verses...")
    for key, gegbe_text in gegbe_index.items():
        if key in ewe_index:
            parallel_pairs.append({
                "mina": gegbe_text,
                "ewe": ewe_index[key]
            })

    logger.info(f"Found {len(parallel_pairs)} parallel verses.")

    if not parallel_pairs:
        logger.warning("No parallel pairs found.")
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mina", "ewe"])
        writer.writeheader()
        writer.writerows(parallel_pairs)

    logger.info(f"Parallel dataset saved to {output_csv}")

if __name__ == "__main__":
    prepare_parallel_dataset()
