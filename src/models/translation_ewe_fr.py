import os
import logging
from transformers import MarianMTModel, MarianTokenizer
import ctranslate2
from src.config.settings import EWE_FR_MODEL, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EweFrenchTranslator:
    def __init__(self, use_ctranslate2=True):
        self.model_name = EWE_FR_MODEL
        self.use_ctranslate = use_ctranslate2
        self.ct_model_path = PROJECT_ROOT / "models" / "ewe_fr_ct2"
        
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        if self.use_ctranslate and self.ct_model_path.exists():
            logger.info(f"Chargement du modèle CTranslate2 depuis {self.ct_model_path}")
            self.translator = ctranslate2.Translator(str(self.ct_model_path), device="cpu")
        else:
            logger.info(f"Chargement du modèle Transformers {self.model_name}")
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.use_ctranslate = False

    def translate(self, text):
        if not text:
            return ""

        if self.use_ctranslate:
            source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            results = self.translator.translate_batch([source])
            target = results[0].hypotheses[0]
            return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target), skip_special_tokens=True)
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True)
            translated = self.model.generate(**inputs)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test simple
    translator = EweFrenchTranslator(use_ctranslate2=False) # On force transformers pour le test
    test_text = "Mose I 1:1" # Juste un test, l'Ewe réel est nécessaire
    print(f"Test translation: {translator.translate('Agbe enye dɔ.')}")
