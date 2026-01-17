# Traducteur et Reconnaissance Vocale Éwé (Speech-to-Text & Translation)

Ce projet vise à développer une solution complète de reconnaissance vocale et de traduction pour la langue **Éwé** (langue vernaculaire parlée principalement au Togo, au Ghana et au Bénin). L'objectif final est de permettre la conversion de la parole en texte (Speech-to-Text) et la traduction bidirectionnelle entre l'Éwé et des langues comme le Français ou l'Anglais, le tout via une interface graphique intuitive.

**Membres du groupe**
- AFOMALE David
- DOGBO Sarah
- BOTRE Aboudou
- TEPE Paulin
- NOYOULIWA Victoire

> [!NOTE]
> **Statut actuel :** Le projet est en cours de développement. Les phases de collecte de données (scraping) et de prétraitement sont opérationnelles. L'entraînement des modèles et le développement de l'interface graphique (GUI) sont les prochaines étapes.

## Vision du Projet

Le système final offrira les fonctionnalités suivantes :
1.  **Reconnaissance Vocale (ASR) :** Conversion de fichiers audio ou de flux vocaux en direct en texte Éwé.
2.  **Traduction Bidirectionnelle :**
    -   Éwé ➔ Français / Anglais
    -   Français / Anglais ➔ Éwé
3.  **Interface Graphique (GUI) :** Une application simple permettant aux utilisateurs d'interagir facilement avec le système.

## Structure du Projet

```text
├── data/
│   ├── raw/                # Données brutes (scripts de scraping)
│   └── processed/          # Données nettoyées et prêtes pour l'entraînement
├── models/                 # Modèles entraînés (Whisper, etc.)
├── notebooks/              # Expérimentations et suivi d'entraînement
├── src/
│   ├── scraping/           # Scripts de collecte de données (Bible Éwé)
│   ├── preprocessing/      # Nettoyage audio et textuel
│   ├── pipeline/           # Orchestration des tâches
│   └── config/             # Paramètres du projet
└── requirements.txt        # Dépendances Python
```

## État d'avancement technique

### 1. Collecte de données (Scraping)
Un scraper robuste a été mis en œuvre pour extraire le corpus biblique Éwé (texte et audio) depuis bible.com. Cela constitue la base de notre dataset d'entraînement.
-   **Script :** `src/scraping/ewe_bible_scraper.py`
-   **Lancement :** `python src/pipeline/build_corpus.py`

### 2. Prétraitement
Les données brutes sont transformées pour être compatibles avec les modèles de Deep Learning (comme Whisper) :
-   **Audio :** Conversion en WAV mono 16kHz (nécessaire pour l'ASR).
-   **Texte :** Nettoyage des caractères spéciaux, suppression des numéros de versets et normalisation.
-   **Dataset :** Génération automatique d'un fichier CSV alignant l'audio et le texte.
-   **Scripts :** `src/preprocessing/audio_processing.py`, `src/preprocessing/text_cleaning.py`, `src/preprocessing/dataset_builder.py`.

### 3. Modèles (À venir)
Nous prévoyons d'utiliser et de fine-tuner le modèle **Whisper de OpenAI** pour la partie reconnaissance vocale. Les notebooks de préparation (`notebooks/02_prepare_asr_dataset.ipynb` et `03_train_whisper_ewe.ipynb`) sont déjà en place.

## Installation

### 1. Prérequis
- Python 3.8+
- Bibliothèques additionnelles (voir `requirements.txt`)

### 2. Mise en place
```bash
# Créer et activer l'environnement virtuel
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation actuelle

Pour générer le corpus de données :
```bash
python src/pipeline/build_corpus.py
```
Cela créera les dossiers `raw/` et `processed/` dans le répertoire `data/`.

## Auteur
**AFOMALE Komi David Frank**
Cours : MTH2321

---
*Ce projet est réalisé dans un cadre académique et de recherche pour la promotion des langues vernaculaires à travers les technologies de l'IA.*
