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
    raw_dirs = {
        "ewe": EWE_RAW_DIR / "texts",
        "gegbe": GEGBE_RAW_DIR / "texts"
    }
    
    for lang, d in raw_dirs.items():
        if not d.exists():
            continue
            
        logger.info(f"Traitement des textes pour : {lang}")
        
        # Groupement par chapitre (ex: gen_01, 1ch_01)
        groups = {}
        for txt_path in d.glob("*.txt"):
            parts = txt_path.stem.split('_')
            if len(parts) >= 2:
                chapter_id = "_".join(parts[:2])
                groups.setdefault(chapter_id, []).append(txt_path)
        
        for chapter_id, paths in groups.items():
            cleaned_path = PROCESSED_TEXT_DIR / f"{chapter_id}.txt"

            # On trie les versets par numéro pour l'ordre correct
            def get_verse_num(p):
                last_part = p.stem.split('_')[-1]
                if last_part.isdigit():
                    return int(last_part)
                return 999  # Fallback

            paths.sort(key=get_verse_num)
            
            all_verses = []
            for p in paths:
                text = p.read_text(encoding="utf-8")
                all_verses.append(clean_text(text))
            
            # Fusion avec un espace
            full_chapter_text = " ".join(v for v in all_verses if v)
            
            if full_chapter_text:
                cleaned_path.write_text(full_chapter_text, encoding="utf-8")
                # logger.info(f"✔ {cleaned_path.name} (fusionné)")

    logger.info(f"Nettoyage et fusion terminés dans {PROCESSED_TEXT_DIR}")

