import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.config.settings import NMT_MODEL_SIZE, NMT_DEVICE, PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinaEweTranslator:
    def __init__(self, model_path=None):
        # Par défaut on charge NLLB-200, mais si on a un modèle fine-tuné localement, on le prend
        self.model_name = model_path if model_path else f"facebook/{NMT_MODEL_SIZE}"
        
        logger.info(f"Chargement du modèle Mina-Ewe : {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(NMT_DEVICE)

    def translate(self, text):
        if not text:
            return ""
        
        # Le Mina et l'Ewe n'ont pas de codes officiels distincts dans NLLB pour le moment
        # On utilise ewe_Latn comme cible. Pour la source, on utilise ewe_Latn ou ace_Latn par défaut
        # Note: Dans un vrai fine-tuning, on peut définir des jetons spéciaux.
        inputs = self.tokenizer(text, return_tensors="pt").to(NMT_DEVICE)
        
        # On force la langue cible à l'Ewe
        translated_tokens = self.model.generate(
            **inputs, 
            forced_bos_token_id=self.tokenizer.lang_code_to_id["ewe_Latn"], 
            max_length=128
        )
        
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

if __name__ == "__main__":
    translator = MinaEweTranslator()
    print(f"Test: {translator.translate('Egbé nyé gbe gba.')}")
