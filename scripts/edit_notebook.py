import json
from pathlib import Path

notebook_path = Path("notebooks/03_train_whisper_ewe.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and any("asr_path = settings.PROJECT_ROOT" in line for line in cell["source"]):
        print("Cellule d'inférence trouvée. Mise à jour des chemins...")
        cell["source"] = [
            "# Chemins locaux\n",
            "asr_path = settings.PROJECT_ROOT / \"models\" / \"whisper-ewe-mina-final\"\n",
            "nllb_path = settings.PROJECT_ROOT / \"models\" / \"nllb-mina-ewe-final\"\n",
            "opus_path = settings.PROJECT_ROOT / \"models\" / \"opus-ewe-fr-local\"\n",
            "\n",
            "print(\"1. Chargement du modèle ASR local...\")\n",
            "try:\n",
            "    loaded_processor = WhisperProcessor.from_pretrained(asr_path)\n",
            "    loaded_model = WhisperForConditionalGeneration.from_pretrained(asr_path)\n",
            "    print(\"   OK.\")\n",
            "except Exception as e:\n",
            "    print(f\"   Erreur ASR (il faut lancer l'entraînement avant) : {e}\")\n",
            "    # Fallback pour ne pas bloquer si pas entraîné dans cette session\n",
            "    loaded_processor = processor \n",
            "    loaded_model = model\n",
            "\n",
            "print(\"2. Initialisation de la Cascade avec les modèles locaux...\")\n",
            "cascade = TranslationCascade(nllb_path=str(nllb_path), opus_path=str(opus_path))\n",
            "print(\"   OK.\")\n"
        ]
        updated = True
        break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook mis à jour avec les nouveaux chemins de modèles.")
else:
    print("Erreur: Cellule d'inférence non trouvée dans le notebook.")
