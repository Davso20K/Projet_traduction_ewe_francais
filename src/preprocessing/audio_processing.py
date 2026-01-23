import subprocess
import imageio_ffmpeg
from pathlib import Path

import logging

from src.config.settings import PROJECT_ROOT, EWE_RAW_DIR, GEGBE_RAW_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# MP3 → WAV mono 16kHz (OBLIGATOIRE pour ASR)

def convert_mp3_to_wav_16k(lang=None):
    """
    Convertit les MP3 en WAV 16kHz mono.
    Si lang est spécifié ("ewe" ou "gegbe"), ne traite que ce dossier.
    """
    # ───────────────────────────────────────────────
    # 1. Déterminer quels dossiers traiter
    # ───────────────────────────────────────────────
    if lang == "ewe":
        raw_dirs = [EWE_RAW_DIR / "audio"]
        forced_lang = "ewe"
    elif lang == "gegbe":
        raw_dirs = [GEGBE_RAW_DIR / "audio"]
        forced_lang = "gegbe"
    elif lang is None:
        raw_dirs = [
            EWE_RAW_DIR / "audio",
            GEGBE_RAW_DIR / "audio"
        ]
        forced_lang = None   # on détectera par dossier si besoin
    else:
        logger.error(f"Langue non reconnue : {lang}")
        return

    # ───────────────────────────────────────────────
    # 2. Logs de diagnostic (à garder ou commenter après debug)
    # ───────────────────────────────────────────────
    logger.info(f"EWE_RAW_DIR/audio existe ? {(EWE_RAW_DIR / 'audio').exists()} — "
                f"{len(list((EWE_RAW_DIR / 'audio').glob('*.mp3')))} mp3")
    logger.info(f"GEGBE_RAW_DIR/audio existe ? {(GEGBE_RAW_DIR / 'audio').exists()} — "
                f"{len(list((GEGBE_RAW_DIR / 'audio').glob('*.mp3')))} mp3")

    total_processed = 0

    for d in raw_dirs:
        if not d.exists():
            logger.warning(f"Dossier introuvable, ignoré : {d}")
            continue

        current_files = list(d.glob("*.mp3"))
        if not current_files:
            logger.warning(f"Aucun fichier .mp3 trouvé dans {d}")
            continue

        # ───────────────────────────────────────────────
        # Détermination fiable du préfixe de langue
        # ───────────────────────────────────────────────
        if forced_lang is not None:
            # On utilise la valeur passée à lang → le cas le plus fréquent et sûr
            current_lang = forced_lang
        else:
            # Fallback rare (appel sans lang) → détection par nom de dossier parent
            parent_name = d.parent.name.lower()
            if "ewe" in parent_name or "ewé" in parent_name:
                current_lang = "ewe"
            elif "gegbe" in parent_name or "gègbe" in parent_name or "gbegbe" in parent_name:
                current_lang = "gegbe"
            else:
                logger.warning(f"Impossible de deviner la langue pour {d} → on skip")
                continue

        logger.info(f"Traitement de {len(current_files)} fichiers pour {current_lang!r}... (dossier: {d})")

        for mp3_path in current_files:
            # Nom avec préfixe langue + nom original sans .mp3
            wav_name = f"{current_lang}_{mp3_path.stem}.wav"
            wav_path = PROCESSED_AUDIO_DIR / wav_name

            if wav_path.exists():
                # logger.debug(f"Déjà existant, ignoré : {wav_path.name}")
                continue

            try:
                subprocess.run([
                    FFMPEG_EXE,
                    "-i", str(mp3_path),
                    "-ar", "16000",
                    "-ac", "1",
                    "-c:a", "pcm_s16le",     # on précise explicitement 16-bit PCM
                    str(wav_path),
                    "-y",
                    "-loglevel", "error"
                ], check=True, capture_output=True)

                logger.info(f"✔ {wav_path.name}")
                total_processed += 1

            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Erreur ffmpeg sur {mp3_path.name} : {e.stderr.decode() if e.stderr else e}")
            except Exception as e:
                logger.error(f"❌ Erreur inattendue sur {mp3_path.name} : {e}")

    logger.info(f"Conversion terminée. {total_processed} nouveaux fichiers créés.")