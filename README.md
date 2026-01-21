# Traducteur et Reconnaissance Vocale (Éwé & Gegbe)

Ce projet vise à développer une solution complète de reconnaissance vocale et de traduction pour les langues **Éwé** et **Gegbe (Mina)** (langues vernaculaires parlées principalement au Togo, au Ghana et au Bénin). L'objectif final est de permettre la conversion de la parole en texte (Speech-to-Text) et la traduction bidirectionnelle entre ces langues et le Français ou l'Anglais, le tout via une interface graphique intuitive.

**Membres du groupe**
- AFOMALE David
- DOGBO Sarah
- BOTRE Aboudou
- TEPE Paulin
- NOYOULIWA Victoire

> [!NOTE]
> **Statut actuel :** L'infrastructure technique est prête pour un workflow **100% Local sur CPU**. Les briques ASR (Whisper) et NMT (Mina ➔ Éwé ➔ Français) sont en place avec des optimisations de quantification (INT8). Le dossier `/models` est utilisé pour stocker les poids locaux et les versions optimisées des modèles.

## Vision du Projet

Le système final offrira les fonctionnalités suivantes :
1.  **Reconnaissance Vocale (ASR) :** Conversion de fichiers audio Mina/Éwé en texte via un modèle Whisper unifié.
2.  **Chaîne de Traduction en Cascade :**
    -   Mina ➔ Éwé (Adaptation dialectale via NLLB fine-tuné).
    -   Éwé ➔ Français / Autre (via Helsinki-NLP Opus-MT).
3.  **Optimisation CPU :** Utilisation de la quantification **INT8**, **faster-whisper** et **CTranslate2** pour garantir des performances fluides sur du matériel grand public (CPU).
4.  **Interface Graphique (GUI) :** Application interactive pour la saisie audio/texte et l'affichage des traductions.

## Structure du Projet

```text
├── data/
│   ├── raw/                # Données brutes de scraping
│   ├── processed/          # Corpus parallèle aligné (Mina/Éwé)
├── models/                 # Stockage local des modèles (Whisper, NLLB, OPUS)
├── notebooks/              # Travaux d'expérimentation
├── src/
│   ├── config/             # Hyperparamètres (batch size, freeze, etc.)
│   ├── models/             # Logique d'inférence et d'entraînement
│   ├── preprocessing/      # Alignement et nettoyage de corpus
│   └── scraping/           # Scripts de collecte Bible
└── requirements.txt        # Dépendances Python
```

## État d'avancement technique

### 1. Collecte de données (Scraping)
Des scrapers robustes et optimisés ont été mis en œuvre pour extraire le corpus biblique complet (73 livres, texte et audio) depuis bible.com.
-   **Éwé :** `src/scraping/ewe_bible_scraper.py` (Optimisé, gestion des reprises).
-   **Gegbe (Mina) :** `src/scraping/gegbe_bible_scraper.py` (Optimisé, gestion des reprises).
-   **Structure :** Extraction d'un fichier audio par chapitre pour une meilleure efficacité de stockage et de parsing.
-   **Lancement :** `python -m src.pipeline.build_corpus`


### 2. Prétraitement
Les données brutes sont transformées pour être compatibles avec les modèles de Deep Learning (comme Whisper) :
-   **Audio :** Conversion en WAV mono 16kHz (nécessaire pour l'ASR).
-   **Texte :** Nettoyage des caractères spéciaux, suppression des numéros de versets et normalisation.
-   **Dataset :** Génération automatique d'un fichier CSV alignant l'audio et le texte.
-   **Scripts :** `src/preprocessing/audio_processing.py`, `src/preprocessing/text_cleaning.py`, `src/preprocessing/dataset_builder.py`.

### 3. Modèles & Orchestration (CPU Ready)
L'intelligence du système est répartie en trois modules pilotables localement :
-   **Config Centralisée :** Tous les réglages se trouvent dans `src/config/settings.py` (gel des couches, learning rate).
-   **Traduction Cascade :** Inférence via `src/pipeline/translate_cascade.py` (Mina ➔ Éwé ➔ Français).
-   **Entraînement Local :** Script `src/models/train_whisper_cpu.py` optimisé pour ne pas saturer le processeur.

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

# Installer les dépendances (IMPORTANT)
pip install -r requirements.txt
```

## Guide d'Utilisation (Pas-à-pas)

Pour mener à bien le projet de la collecte à la traduction finale, suivez ces étapes dans l'ordre :

### Étape 1 : Collecte des données
Lancez le scraper pour récupérer les textes et audios des bibles Éwé et Mina.
```bash
python -m src.pipeline.build_corpus
```
*Cette étape crée aussi automatiquement le fichier d'alignement `data/processed/parallel_mina_ewe.csv`.*

### Étape 2 : Préparation du Dataset ASR
Ouvrez et exécutez le notebook **`notebooks/02_prepare_asr_dataset.ipynb`**. 
- Il convertira les audios en WAV 16kHz.
- Il nettoiera les textes.
- Il générera le dataset final pour l'entraînement.

### Étape 3 : Entraînement Local (CPU)
Ouvrez le notebook **`notebooks/03_train_whisper_ewe.ipynb`**.
- Il appelle le module `src.models.train_whisper_cpu`.
- Vous pouvez y modifier les hyperparamètres (gel des couches, batch size) et suivre l'avancement de l'apprentissage ASR.

### Étape 4 : Traduction Finale (Cascade)
Utilisez le même notebook ou le terminal pour tester la chaîne complète :
```bash
python -m src.pipeline.translate_cascade
```
*Le système prendra une phrase en Mina, la pivotera en Éwé, puis la traduira en Français.*

## Configuration du Matériel
Le projet est optimisé pour tourner sur **CPU uniquement**. 
- Inférence : **INT8** via CTranslate2/faster-whisper.
- Entraînement : **Paramètres gelés à 90%** pour économiser la RAM et le processeur.


## Auteur
**AFOMALE Komi David Frank**
Cours : MTH2321

---
*Ce projet est réalisé dans un cadre académique et de recherche pour la promotion des langues vernaculaires à travers les technologies de l'IA.*
