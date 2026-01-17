from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ewe_bible"

AUDIO_DIR = RAW_DATA_DIR / "audio"
TEXT_DIR = RAW_DATA_DIR / "texts"
META_DIR = RAW_DATA_DIR / "metadata"
