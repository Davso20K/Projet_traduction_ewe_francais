import subprocess
import imageio_ffmpeg
from pathlib import Path
import torchaudio
import logging

from src.config.settings import AUDIO_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

#MP3 → WAV mono 16kHz (OBLIGATOIRE pour ASR)

def convert_mp3_to_wav_16k():
    audio_files = list(AUDIO_DIR.glob("*.mp3"))

    logger.info(f"{len(audio_files)} fichiers audio trouvés")

    for mp3_path in audio_files:
        wav_path = PROCESSED_AUDIO_DIR / mp3_path.with_suffix(".wav").name

        if wav_path.exists():
            continue

        try:
            # Utilisation directe de ffmpeg pour une conversion robuste et rapide
            # -ar 16000 : fréquence d'échantillonnage 16kHz
            # -ac 1 : conversion en mono
            subprocess.run([
                FFMPEG_EXE, 
                "-i", str(mp3_path), 
                "-ar", "16000", 
                "-ac", "1", 
                str(wav_path), 
                "-y", 
                "-loglevel", "error"
            ], check=True)
            
            logger.info(f"✔ {wav_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Erreur lors de la conversion de {mp3_path.name}: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur inattendue pour {mp3_path.name}: {e}")
