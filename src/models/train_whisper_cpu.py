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
    TRAINING_NUM_CORES,
)

import os
import multiprocessing

# --- Optimisation CPU ---
# Sur Windows, set_num_threads est utile pour MKL/OpenMP
NUM_CORES = TRAINING_NUM_CORES  # Utilise la limite de 10 cœurs demandée
torch.set_num_threads(NUM_CORES) 

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
    model.config.use_cache = False  # DOIT être False si gradient_checkpointing est True, et recommandé sur CPU

    # 1. Chargement dataset
    if dataset is None:
        csv_path = PROCESSED_DIR / "bible_asr_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {csv_path}. Veuillez relancer `dataset_builder.py`.")

        print(f"Chargement depuis {csv_path}...")
        dataset = load_dataset("csv", data_files={"train": str(csv_path)})

        # Subsampling for RAM safety if needed
        from src.config.settings import ASR_MAX_SAMPLES
        if ASR_MAX_SAMPLES and len(dataset["train"]) > ASR_MAX_SAMPLES:
            print(f"Subsampling dataset from {len(dataset['train'])} to {ASR_MAX_SAMPLES} samples for system stability...")
            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(ASR_MAX_SAMPLES))

    # On n'utilise PLUS cast_column avec Audio() car cela déclenche torchcodec qui plante sur votre machine.
    # dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    # 2. Pré-traitement simple avec chargement manuel (SoundFile)
    import soundfile as sf
    import scipy.signal
    import numpy as np

    def prepare_dataset(batch):
        # Chargement manuel pour éviter l'erreur torchcodec
        audio_path = batch["audio_filepath"]
        
        try:
            waveform, sr = sf.read(audio_path)
        except Exception as e:
            # En cas d'erreur de lecture, on retourne des dummy data pour ne pas crasher tout le process
            # (idéalement on filtrerait avant, mais map gère mal les suppressions directes)
            print(f"Erreur lecture {audio_path}: {e}")
            waveform = np.zeros(16000) # 1 sec silence
            sr = 16000

        # Conversion Mono
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample si nécessaire (sécurité)
        if sr != 16000:
            num_samples = int(len(waveform) * 16000 / sr)
            waveform = scipy.signal.resample(waveform, num_samples)

        # Audio input
        input_features = processor.feature_extractor(
            waveform, 
            sampling_rate=16000
        ).input_features[0]

        # Text targets
        lang = batch["language"].lower()
        proxy_lang = "yoruba" if lang in ["ewe", "gegbe", "mina", "gbe"] else lang
        
        # Tokenize text
        # Whisper gère le language via le tokenizer/processor lors du forward, 
        # mais on peut aussi forcer le prompt ici si on veut
        labels = processor.tokenizer(batch["text"]).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels,
        }

    print("Pré-traitement du dataset (Feature Extraction manuelle)...")
    # On utilise num_proc pour paralléliser le prétraitement
    dataset = dataset.map(
        prepare_dataset, 
        remove_columns=dataset["train"].column_names, 
        num_proc=NUM_CORES
    )

    print(f"Dataset transformé : {len(dataset['train'])} exemples.")

    # 3. Filtrage des séquences trop longues (décodeur Whisper limité à 448 tokens)
    MAX_LABEL_LENGTH = 448
    def filter_labels(labels):
        return len(labels) <= MAX_LABEL_LENGTH

    print(f"Filtrage des labels > {MAX_LABEL_LENGTH} tokens...")
    dataset = dataset.filter(
        lambda x: filter_labels(x["labels"]), 
        num_proc=NUM_CORES
    )
    print(f"Dataset final : {len(dataset['train'])} exemples.")

    # 4. Split
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
        gradient_accumulation_steps=1,             # Désactivé pour debug autograd
        learning_rate=ASR_LEARNING_RATE,
        warmup_steps=50,
        max_steps=1000,                            # Plus de steps car données plus petites/nombreuses
        gradient_checkpointing=False,              # Désactivé car cause RuntimeError sur CPU
        fp16=False,                                # CPU ne supporte pas fp16 (ou mal), bfloat16 si supporté
        eval_strategy="steps",
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
    return model, processor

if __name__ == "__main__":
    train_whisper_on_cpu()