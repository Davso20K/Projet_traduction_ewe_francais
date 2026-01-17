from pathlib import Path
import torchaudio
import logging

from src.config.settings import AUDIO_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_AUDIO_DIR = PROJECT_ROOT / "data" / "processed" / "audio_16k"
PROCESSED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

#MP3 → WAV mono 16kHz (OBLIGATOIRE pour ASR)

def convert_mp3_to_wav_16k():
    audio_files = list(AUDIO_DIR.glob("*.mp3"))

    logger.info(f"{len(audio_files)} fichiers audio trouvés")

    for mp3_path in audio_files:
        wav_path = PROCESSED_AUDIO_DIR / mp3_path.with_suffix(".wav").name

        if wav_path.exists():
            continue

        waveform, sr = torchaudio.load(mp3_path)

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=16000
            )

        torchaudio.save(wav_path, waveform, 16000)
        logger.info(f"✔ {wav_path.name}")
