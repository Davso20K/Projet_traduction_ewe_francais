import json
import csv
from pathlib import Path
import logging

from src.config.settings import PROJECT_ROOT, META_DIR, GEGBE_META_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIO_DIR_16K = PROCESSED_DIR / "audio_16k"
TEXT_DIR_CLEAN = PROCESSED_DIR / "transcripts"

# On peut soit fusionner, soit séparer. Ici on fusionne dans ewe_asr_dataset.csv 
# ou on en crée un global. Créons un global "bible_asr_dataset.csv"
OUTPUT_CSV = PROCESSED_DIR / "bible_asr_dataset.csv"

def build_asr_dataset():
    meta_files = [
        META_DIR / "ewe_bible_raw.json",
        GEGBE_META_DIR / "gegbe_bible_raw.json"
    ]

    rows = []
    processed_audio = set()

    for meta_path in meta_files:
        if not meta_path.exists():
            logger.warning(f"Fichier meta introuvable : {meta_path}")
            continue
            
        lang = "ewe" if "ewe" in meta_path.name else "gegbe"
        records = json.loads(meta_path.read_text(encoding="utf-8"))

        for r in records:
            if not r["audio_path"]:
                continue
            
            audio_id = r["audio_path"]
            if audio_id in processed_audio:
                continue

            # Construction des noms de fichiers avec prefixe langue (pour correspondre à audio_processing.py et text_cleaning.py)
            wav_name = f"{lang}_{Path(audio_id).with_suffix('.wav').name}"
            txt_name = f"{lang}_{Path(audio_id).with_suffix('.txt').name}"
            
            wav_path = AUDIO_DIR_16K / wav_name
            txt_path = TEXT_DIR_CLEAN / txt_name
            
            if not wav_path.exists():
                # Check legacy filename
                legacy_wav = AUDIO_DIR_16K / Path(audio_id).with_suffix('.wav').name
                if legacy_wav.exists():
                     logger.warning(f"Found legacy file {legacy_wav.name} but missing {wav_name}. PROCESSING STEP REQUIRED: Please restart kernel and re-run audio processing.")
                elif len(processed_audio) < 5:
                    logger.debug(f"Missing: wav={wav_path.exists()}, txt={txt_path.exists()} for {wav_name}")
                continue

            text = txt_path.read_text(encoding="utf-8")

            if len(text) < 10: # Versets ou chapitres très courts ignorés
                continue

            rows.append({
                "audio_filepath": str(wav_path),
                "text": text,
                "language": lang
            })
            processed_audio.add(audio_id)

    if not rows:
        logger.warning("Aucune donnée trouvée pour le dataset ASR")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_filepath", "text", "language"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Dataset ASR créé : {len(rows)} lignes ({OUTPUT_CSV.name})")

