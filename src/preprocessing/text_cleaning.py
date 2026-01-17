import re
from pathlib import Path
import logging

from src.config.settings import TEXT_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "transcripts"
PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Nettoyage du texte Ewe (versets propres)
def clean_ewe_text(text: str) -> str:
    text = text.strip()

    # supprimer numéros de versets en début
    text = re.sub(r"^\d+\s*", "", text)

    # espaces multiples
    text = re.sub(r"\s+", " ", text)

    # garder alphabet + diacritiques
    text = re.sub(r"[^\w\sàáãèéɛìíòóɔùúɖŋ']", "", text)

    return text


def clean_all_texts():
    for txt_path in TEXT_DIR.glob("*.txt"):
        cleaned_path = PROCESSED_TEXT_DIR / txt_path.name

        if cleaned_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        cleaned = clean_ewe_text(text)

        cleaned_path.write_text(cleaned, encoding="utf-8")
        logger.info(f"✔ {cleaned_path.name}")
