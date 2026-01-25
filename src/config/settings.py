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

# Processed Data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters (CPU Optimized)
ASR_MODEL_SIZE = "base"  # Options: tiny, base, small
ASR_FREEZE_PERCENT = 0.9  # Freeze 90% of parameters
ASR_LEARNING_RATE = 1e-4
ASR_BATCH_SIZE = 4
ASR_EPOCHS = 10
TRAINING_NUM_CORES = 4  # Réduit de 10 à 4 pour économiser la RAM sur PC modeste
ASR_MAX_SAMPLES = 5000   # Limite optionnelle pour éviter les crashs si le dataset est trop gros

# NMT Hyperparameters
NMT_MODEL_SIZE = "nllb-200-distilled-600M"
NMT_QUANTIZATION = "int8"
NMT_DEVICE = "cpu"

# External Models
EWE_FR_MODEL = "Helsinki-NLP/opus-mt-ee-fr"

