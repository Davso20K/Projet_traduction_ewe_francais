import json
from pathlib import Path

notebook_path = Path("notebooks/03_train_whisper_ewe.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define new cells
new_cells = [
    {
        "cell_type": "markdown",
        "id": "nmt_train_header",
        "metadata": {},
        "source": [
            "## 2. Amélioration de la Traduction (Mina -> Éwé)\n",
            "\n",
            "Le modèle NLLB-200 par défaut peut manquer de précision pour les spécificités locales. Nous fine-tunons ici le traducteur sur le corpus parallèle de la Bible (Mina/Éwé)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "nmt_data_prep",
        "metadata": {},
        "outputs": [],
        "source": [
            "import importlib\n",
            "import src.preprocessing.prepare_nmt_dataset\n",
            "importlib.reload(src.preprocessing.prepare_nmt_dataset)\n",
            "from src.preprocessing.prepare_nmt_dataset import prepare_parallel_dataset\n",
            "\n",
            "print(\"Préparation du dataset parallèle Mina-Éwé...\")\n",
            "prepare_parallel_dataset()\n",
            "print(\"Terminé.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "nmt_train_run",
        "metadata": {},
        "outputs": [],
        "source": [
            "import src.models.train_translation_cpu\n",
            "importlib.reload(src.models.train_translation_cpu)\n",
            "from src.models.train_translation_cpu import train_mina_ewe_nmt\n",
            "\n",
            "print(\"Démarrage du fine-tuning du traducteur (CPU)...\")\n",
            "# Cet entraînement est beaucoup plus rapide que Whisper.\n",
            "train_mina_ewe_nmt()\n",
            "print(\"Entraînement NMT terminé !\")"
        ]
    }
]

# Find index to insert (after section 1)
insert_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and any("## 2." in line for line in cell["source"]):
        # Update renumbered sections
        cell["source"] = [line.replace("## 2.", "## 3.") for line in cell["source"]]
        if insert_idx == -1:
            insert_idx = i
    elif cell["cell_type"] == "markdown" and any("## 3." in line for line in cell["source"]):
         cell["source"] = [line.replace("## 3.", "## 4.") for line in cell["source"]]

if insert_idx != -1:
    nb["cells"][insert_idx:insert_idx] = new_cells
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook mis à jour avec succès.")
else:
    print("Erreur: Section '## 2.' non trouvée.")
