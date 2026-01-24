import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
import evaluate
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

import os
import multiprocessing

# --- Optimisation CPU ---
# Sur Windows, set_num_threads est utile pour MKL/OpenMP
NUM_CORES = multiprocessing.cpu_count()
torch.set_num_threads(max(1, NUM_CORES - 2)) 

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


wer_metric = evaluate.load("wer")

def train_whisper_on_cpu(dataset=None):
    """
    Fine-tune Whisper sur le dataset pré-aligné (verset par verset).
    Optimisé pour CPU.
    """
    model_name = f"openai/whisper-{ASR_MODEL_SIZE}"
    print(f"--- Initialisation Fine-tuning Whisper ({model_name}) sur CPU ({NUM_CORES} coeurs) ---")

    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # 1. Chargement dataset
    if dataset is None:
        csv_path = PROCESSED_DIR / "bible_asr_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {csv_path}. Veuillez relancer `dataset_builder.py`.")

        print(f"Chargement depuis {csv_path}...")
        dataset = load_dataset("csv", data_files={"train": str(csv_path)})

    # Important: Cast de la colonne audio pour chargement lazy automatique via datasets
    # Cela permet à datasets de charger le WAV automatiquement en array
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    # 2. Pré-traitement simple (plus de slicing complexe)
    def prepare_dataset(batch):
        audio = batch["audio_filepath"]
        
        # Audio input
        input_features = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Text targets
        # Gestion langue proxy si besoin (Yoruba est souvent utilisé pour Ewe/Fon/Gbe)
        lang = batch["language"].lower()
        proxy_lang = "yoruba" if lang in ["ewe", "gegbe", "mina", "gbe"] else lang
        
        # Set language token
        dataset_info_ids = processor.tokenizer.get_decoder_prompt_ids(
            language=proxy_lang,
            task="transcribe"
        )
        
        # Tokenize text
        labels = processor.tokenizer(batch["text"]).input_ids
        
        # Combine prompt + text (si processor ne le fait pas auto via language ID,
        # mais WhisperProcessor gère souvent le forced_decoder_ids dans le model.
        # Ici on suit la méthode standard HF fine-tuning)
        
        return {
            "input_features": input_features,
            "labels": labels, # Pas besoin d'ajouter prompt_ids manuellement si on configure le modèle/tokenizer correctement, 
                              # mais pour être sûr on pourrait le faire. Whisper standard gère ça via config.
                              # Pour fine-tuning multilingue explicite, on laisse souvent le tokenizer faire.
        }

    print("Pré-traitement du dataset (Feature Extraction)...")
    # On utilise num_proc pour paralléliser le prétraitement
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names, 
        num_proc=max(1, NUM_CORES - 2)
    )

    print(f"Dataset prêt : {len(dataset['train'])} exemples.")

    # 3. Split
    if "test" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 4. Métriques
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # 5. Training Args CPU Optimized
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "whisper-ewe-gegbe-local"),
        per_device_train_batch_size=ASR_BATCH_SIZE, # 4 ou 8 selons RAM
        gradient_accumulation_steps=4,             # Accumuler pour simuler plus grand batch
        learning_rate=ASR_LEARNING_RATE,
        warmup_steps=50,
        max_steps=1000,                            # Plus de steps car données plus petites/nombreuses
        gradient_checkpointing=True,               # Save RAM
        fp16=False,                                # CPU ne supporte pas fp16 (ou mal), bfloat16 si supporté
        evaluation_strategy="steps",
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=200,
        eval_steps=200,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        use_cpu=True,                              # Force CPU explicitly if needed
        dataloader_num_workers=4,                  # Data loading parallélisé
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Démarrage de l'entraînement...")
    trainer.train()
    
    final_path = PROJECT_ROOT / "models" / "whisper-ewe-gegbe-final"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    print(f"Modèle sauvegardé : {final_path}")

if __name__ == "__main__":
    train_whisper_on_cpu()