import json
import csv
from pathlib import Path
import logging

from src.config.settings import PROJECT_ROOT, META_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIO_DIR_16K = PROCESSED_DIR / "audio_16k"
TEXT_DIR_CLEAN = PROCESSED_DIR / "transcripts"

OUTPUT_CSV = PROCESSED_DIR / "ewe_asr_dataset.csv"

# LE SCRIPT CLÉ pour l’ASR
def build_asr_dataset():
    meta_path = META_DIR / "ewe_bible_raw.json"

    records = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = []

    for r in records:
        if not r["audio_path"]:
            continue

        wav_name = Path(r["audio_path"]).with_suffix(".wav").name
        txt_name = Path(r["audio_path"]).with_suffix(".txt").name

        wav_path = AUDIO_DIR_16K / wav_name
        txt_path = TEXT_DIR_CLEAN / txt_name

        if not wav_path.exists() or not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")

        if len(text) < 5:
            continue

        rows.append({
            "audio_filepath": str(wav_path),
            "text": text
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Dataset ASR créé : {len(rows)} lignes")
