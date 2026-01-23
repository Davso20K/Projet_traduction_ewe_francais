import logging
from src.models.translation_mina_ewe import MinaEweTranslator
from src.models.translation_ewe_fr import EweFrenchTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationCascade:
    def __init__(self, nllb_path=None, opus_path=None):
        logger.info("Initialisation de la cascade de traduction...")
        self.mina_ewe = MinaEweTranslator(model_path=nllb_path)
        self.ewe_fr = EweFrenchTranslator(use_ctranslate2=False, model_path=opus_path) # Fallback auto si pas converti

    def translate_mina_to_french(self, mina_text):
        logger.info(f"Source (Mina): {mina_text}")
        
        # 1. Mina -> Ewe
        ewe_text = self.mina_ewe.translate(mina_text)
        logger.info(f"Pivot (Ewe): {ewe_text}")
        
        # 2. Ewe -> French
        french_text = self.ewe_fr.translate(ewe_text)
        logger.info(f"Cible (Français): {french_text}")
        
        return {
            "mina": mina_text,
            "ewe": ewe_text,
            "french": french_text
        }

if __name__ == "__main__":
    cascade = TranslationCascade()
    result = cascade.translate_mina_to_french("Egbé nyé gbe gba.")
    print("\n--- RÉSULTAT FINAL ---")
    print(f"Mina: {result['mina']}")
    print(f"Ewe: {result['ewe']}")
    print(f"Français: {result['french']}")
