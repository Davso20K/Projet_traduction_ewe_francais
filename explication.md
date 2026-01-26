# Rapport Technique : Système de Traduction de Parole Cascade (Mina -> Français)

Ce document présente les aspects techniques et les choix d'implémentation du projet de traduction automatique de la parole du Mina (Gegbe) vers le Français, en passant par l'Éwé comme langue pivot.

## 1. Acquisition et Préparation des Données

La qualité d'un modèle de Deep Learning dépend de la pertinence de ses données. Pour ce projet, nous avons construit un pipeline d'acquisition robuste :

- **Source de données** : Le corpus provient de [Bible.com](https://bible.com), offrant des textes structurés et des audios de haute qualité pour les langues cibles (Éwé et Gegbe).
-  **Pipeline ASR (Speech-to-Text)** :
    - Extraction automatisée (Scraping) des fichiers MP3 et des textes associés.
    - **Alignement temporel** : Utilisation d'un algorithme d'alignement pour découper les longs audios en segments correspondant précisément à chaque verset.
    - **Volume** : ~23 559 segments audio-texte.
- **Pipeline NMT (Traduction)** :
    - Alignement des versets par ID (Livre/Chapitre/Verset) pour créer un corpus parallèle Mina-Éwé.
    - **Volume** : Corpus complet de la Bible (~31 000 versets).

## 2. Architecture des Modèles

Le système repose sur une **cascade de trois modèles spécialisés**, optimisés pour fonctionner efficacement même sur CPU.

| Composant | Modèle de base | Rôle | Architecture (Couches) |
| :--- | :--- | :--- | :--- |
| **ASR** | `Whisper-base` | Transcription du Mina/Éwé | 6 couches Encoder / 6 couches Decoder |
| **Traduction 1** | `NLLB-200` | Traduction Mina ➔ Éwé | 12 couches Encoder / 12 couches Decoder |
| **Traduction 2** | `Opus-MT` | Traduction Éwé ➔ Français | 6 couches Encoder / 6 couches Decoder |

### Pourquoi ces choix ?
- **Whisper (OpenAI)** : Modèle SOTA (State-of-the-Art) pour la parole, extrêmement robuste aux accents et aux bruits de fond. La version `base` (74M paramètres) offre le meilleur compromis entre précision et vitesse sur CPU.
- **NLLB-200 (Meta)** : Spécialisé dans les langues peu dotées (200+ langues dont l'Éwé), indispensable pour gérer les spécificités du Mina et de l'Éwé.
- **Opus-MT (Helsinki-NLP)** : Modèle léger et performant, spécifiquement entraîné sur le couple Éwé-Français.

## 3. Stratégie d'Entraînement et Optimisation CPU

L'entraînement de modèles de transformer est normalement gourmand en ressources GPU. Pour permettre une exécution sur machine locale, nous avons implémenté les stratégies suivantes :

### Gel des Couches (Freeze)
Pour réduire l'empreinte mémoire et accélérer l'entraînement, nous ne mettons à jour qu'une petite fraction des paramètres :
- **Whisper** : **90% des couches sont gelées**. Nous n'entraînons que les dernières couches du décodeur pour adapter le modèle à la phonétique spécifique des langues Gbe.
- **NMT (NLLB)** : **95% des couches sont gelées**. Le modèle possède déjà une base solide en Éwé ; le fine-tuning se concentre sur l'alignement sémantique Mina-Éwé.

### Optimisations Système
- **Quantification int8** : Les poids des modèles sont réduits de 32 bits à 8 bits pour l'inférence, divisant par 4 la RAM nécessaire sans perte majeure de précision.
- **Gestion des Threads** : Limitation à 4 cœurs CPU (`torch.set_num_threads(4)`) pour éviter la saturation du système et les crashs de RAM.
- **Subsampling** : Limitation optionnelle à 5 000 échantillons pour les phases de test rapide.

## 4. Résultats et Pipeline Final (Orchestration)

Le projet utilise la classe `TranslationCascade` pour orchestrer le flux :
1. **Audio** ➔ `Whisper` ➔ **Texte (Mina)**
2. **Texte (Mina)** ➔ `NLLB-200` ➔ **Texte (Éwé)**
3. **Texte (Éwé)** ➔ `Opus-MT` ➔ **Texte (Français)**

Cette approche modulaire permet d'améliorer chaque composant indépendamment tout en garantissant une traduction finale contextuellement riche.
