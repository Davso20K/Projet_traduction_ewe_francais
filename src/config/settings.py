from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ewe Data
EWE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ewe"
AUDIO_DIR = EWE_RAW_DIR / "audio"
TEXT_DIR = EWE_RAW_DIR / "texts"
META_DIR = EWE_RAW_DIR / "metadata"

# Gegbe (Mina) Data
GEGBE_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gegbe"
GEGBE_AUDIO_DIR = GEGBE_RAW_DIR / "audio"
GEGBE_TEXT_DIR = GEGBE_RAW_DIR / "texts"
GEGBE_META_DIR = GEGBE_RAW_DIR / "metadata"

