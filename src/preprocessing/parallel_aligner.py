import json
import pandas as pd
import logging
from pathlib import Path
from src.config.settings import META_DIR, GEGBE_META_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelAligner:
    def __init__(self):
        self.ewe_meta_path = META_DIR / "ewe_bible_raw.json"
        self.gegbe_meta_path = GEGBE_META_DIR / "gegbe_bible_raw.json"
        self.output_csv = PROCESSED_DIR / "parallel_mina_ewe.csv"

    def load_data(self, path):
        if not path.exists():
            logger.warning(f"Fichier non trouvé: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def align(self):
        logger.info("Chargement des données Ewe...")
        ewe_data = self.load_data(self.ewe_meta_path)
        logger.info("Chargement des données Gegbe (Mina)...")
        gegbe_data = self.load_data(self.gegbe_meta_path)

        if not ewe_data or not gegbe_data:
            logger.error("Impossible d'aligner : une des sources de données est vide.")
            return

        # Création de DataFrames
        df_ewe = pd.DataFrame(ewe_data)
        df_gegbe = pd.DataFrame(gegbe_data)

        # Création d'une clé unique pour l'alignement
        # Clé : BOOK.CHAPTER.VERSE
        df_ewe["verse_id"] = df_ewe.apply(lambda x: f"{x['book']}.{x['chapter']}.{x['verse']}", axis=1)
        df_gegbe["verse_id"] = df_gegbe.apply(lambda x: f"{x['book']}.{x['chapter']}.{x['verse']}", axis=1)

        # Merge sur l'ID de verset
        logger.info("Alignement des versets...")
        merged_df = pd.merge(
            df_gegbe[["verse_id", "text", "audio_path"]],
            df_ewe[["verse_id", "text", "audio_path"]],
            on="verse_id",
            suffixes=("_mina", "_ewe")
        )

        # Nettoyage : suppression des lignes sans texte dans l'un ou l'autre
        merged_df = merged_df.dropna(subset=["text_mina", "text_ewe"])
        
        logger.info(f"Alignement terminé : {len(merged_df)} versets parallèles trouvés.")
        
        # Sauvegarde
        merged_df.to_csv(self.output_csv, index=False, encoding="utf-8")
        logger.info(f"Dataset parallèle sauvegardé dans : {self.output_csv}")
        
        return merged_df

if __name__ == "__main__":
    aligner = ParallelAligner()
    aligner.align()
