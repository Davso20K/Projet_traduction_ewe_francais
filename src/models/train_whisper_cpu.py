import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
import evaluate  # pip install evaluate
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from src.config.settings import (
    ASR_MODEL_SIZE,
    ASR_FREEZE_PERCENT,
    ASR_LEARNING_RATE,
    ASR_BATCH_SIZE,
    PROJECT_ROOT,
    PROCESSED_DIR,
)

import soundfile as sf
import scipy.signal
import numpy as np
import os

# --- DATA COLLATOR (inchangé, mais très important pour Whisper) ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# --- Métrique WER ---
wer_metric = evaluate.load("wer")




def train_whisper_on_cpu(dataset=None):
    """
    Fine-tune Whisper sur dataset local (ewe + gegbe).
    Gère les audios longs via découpage en segments ~28s.
    Utilise la colonne 'language' pour conditionner chaque exemple.
    """
    model_name = f"openai/whisper-{ASR_MODEL_SIZE}"
    print(f"--- Initialisation Fine-tuning Whisper ({model_name}) ---")

    # 1. Chargement processeur et modèle
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # On laisse le modèle détecter la langue (ou forcer via prompt_ids si besoin)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 2. Chargement dataset
    if dataset is None:
        csv_path = PROCESSED_DIR / "bible_asr_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

        print(f"Chargement depuis {csv_path}...")
        dataset = load_dataset("csv", data_files={"train": str(csv_path)})

    # 3. Pré-traitement : découpage longs audios + gestion language
    print("Pré-traitement audio + texte (découpage segments + condition language)...")

    MAX_SEGMENT_S = 28  # secondes — sécurité pour Whisper
    MAX_SAMPLES = MAX_SEGMENT_S * 16000

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Remplace -100 par le pad_token_id du tokenizer de Whisper
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Décodage avec le tokenizer du processeur (qui connaît bien Whisper)
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Calcul du WER avec evaluate
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def prepare_dataset(examples):
        """
        Version batched pour découpage : on retourne des listes de features/labels
        """
        audio_paths = examples["audio_filepath"]
        texts = examples["text"]
        langs = examples["language"]

        all_input_features = []
        all_labels = []

        for audio_path, full_text, lang_str in zip(audio_paths, texts, langs):
            lang = lang_str.lower()

            # Proxy
            proxy_lang = lang
            if lang in ["ewe", "gegbe", "gègbè", "gegbè", "mina", "gbe"]:
                proxy_lang = "yoruba"

            if not os.path.exists(audio_path):
                continue  # skip sans erreur

            waveform, sr = sf.read(audio_path)
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            if sr != 16000:
                waveform = scipy.signal.resample(waveform, int(len(waveform) * 16000 / sr))

            prompt_ids = list(processor.get_decoder_prompt_ids(
                language=proxy_lang,
                task="transcribe"
            )[0])

            start = 0
            while start < len(waveform):
                end = min(start + MAX_SAMPLES, len(waveform))
                segment_wave = waveform[start:end]

                input_features = processor.feature_extractor(
                    segment_wave,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features[0]

                tokenized_text = processor.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=448 - len(prompt_ids),
                    return_tensors="pt"
                )

                text_ids = tokenized_text["input_ids"][0].tolist()
                labels = prompt_ids + text_ids
                if len(labels) > 448:
                    labels = labels[:448]

                all_input_features.append(input_features)
                all_labels.append(labels)

                start = end

        # On retourne un dict avec des listes → datasets va flatten automatiquement
        return {
            "input_features": all_input_features,
            "labels": all_labels,
        }
    # Appliquer le mapping (sans multiprocessing sur CPU)
    dataset = dataset.map(
        prepare_dataset,
        batched=True,           # ← maintenant batched
        batch_size=4,           # traite 4 exemples à la fois (ajuste selon RAM)
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    
    print(f"Dataset après préparation : {len(dataset['train'])} exemples")

    # 4. Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 5. Gel partiel (inchangé)
    num_parameters = sum(p.numel() for p in model.parameters())
    freeze_limit = int(num_parameters * ASR_FREEZE_PERCENT)
    print(f"Modèle : {num_parameters:,} params → gel de {ASR_FREEZE_PERCENT*100:.0f}%")

    current_params = 0
    for param in model.parameters():
        if current_params < freeze_limit:
            param.requires_grad = False
        current_params += param.numel()

    # 6. Arguments d'entraînement (améliorés)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "whisper-ewe-gegbe-local"),
        per_device_train_batch_size=ASR_BATCH_SIZE,
        gradient_accumulation_steps=4,           # augmenté pour CPU
        learning_rate=ASR_LEARNING_RATE,
        warmup_steps=50,
        max_steps=500,                           # ← plus long que 50, mais raisonnable pour test
        # num_train_epoch=3,                    # alternative aux steps
        fp16=False,
        predict_with_generate=True,              # nécessaire pour WER
        generation_max_length=448,
        save_steps=100,
        eval_steps=50,
        eval_strategy="steps",
        logging_steps=  10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",             # ← on optimise sur WER maintenant
        greater_is_better=False,
        gradient_checkpointing=True,             # économise mémoire sur CPU
        push_to_hub=False,
        use_cpu=True,
        remove_unused_columns=False,
    )

    # 7. Split train/test si absent
    if "test" not in dataset:
        print("Pas de split test → création 90/10...")
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = split

    # 8. Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Démarrage du fine-tuning...")
    trainer.train()
    print("Entraînement terminé.")

    # Sauvegarde finale
    final_path = PROJECT_ROOT / "models" / "whisper-ewe-gegbe-final"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"Modèle sauvegardé dans : {final_path}")

    return model, processor


if __name__ == "__main__":
    train_whisper_on_cpu()