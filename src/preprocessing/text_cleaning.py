import re
from pathlib import Path
import logging

from src.config.settings import PROJECT_ROOT, EWE_RAW_DIR, GEGBE_RAW_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "transcripts"
PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Nettoyage du texte (versets propres) - Partagé par Ewe et Gegbe
def clean_text(text: str) -> str:
    text = text.strip()

    # supprimer numéros de versets en début
    text = re.sub(r"^\d+\s*", "", text)

    # espaces multiples
    text = re.sub(r"\s+", " ", text)

    # garder alphabet + diacritiques (L'alphabet Gegbe est proche de l'Ewe)
    text = re.sub(r"[^\w\sàáãèéɛìíòóɔùúɖŋ']", "", text)

    return text


def clean_all_texts():
    raw_dirs = [
        EWE_RAW_DIR / "texts",
        GEGBE_RAW_DIR / "texts"
    ]
    
    txt_paths = []
    for d in raw_dirs:
        if d.exists():
            txt_paths.extend(list(d.glob("*.txt")))

    for txt_path in txt_paths:
        cleaned_path = PROCESSED_TEXT_DIR / txt_path.name

        if cleaned_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        cleaned = clean_text(text)

        cleaned_path.write_text(cleaned, encoding="utf-8")
        logger.info(f"✔ {cleaned_path.name}")

