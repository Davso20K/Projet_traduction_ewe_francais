import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.config.settings import ASR_MODEL_SIZE, ASR_FREEZE_PERCENT, ASR_LEARNING_RATE, ASR_BATCH_SIZE, ASR_EPOCHS, PROJECT_ROOT

def train_whisper_on_cpu(dataset):
    model_name = f"openai/whisper-{ASR_MODEL_SIZE}"
    print(f"Chargement du modèle {model_name}...")
    
    processor = WhisperProcessor.from_pretrained(model_name, language="ewe", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Stratégie de gel des couches (Freeze) pour CPU
    # On gèle la majorité du modèle pour ne pas saturer le CPU et la RAM
    num_parameters = sum(p.numel() for p in model.parameters())
    freeze_limit = int(num_parameters * ASR_FREEZE_PERCENT)
    
    print(f"Extraction de {num_parameters} paramètres. Gel de {ASR_FREEZE_PERCENT*100}%...")
    
    # Simple freeze des paramètres (approximation par index)
    current_params = 0
    for param in model.parameters():
        if current_params < freeze_limit:
            param.requires_grad = False
        current_params += param.numel()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "whisper-ewe-mina-local"),
        per_device_train_batch_size=ASR_BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=ASR_LEARNING_RATE,
        warmup_steps=50,
        max_steps=500, # On limite pour un test CPU
        fp16=False, # Pas de fp16 sur CPU standard
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=100,
        eval_steps=100,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        no_cuda=True, # Important pour CPU
    )

    # Le trainer serait initialisé ici avec le dataset préparé
    # trainer = Seq2SeqTrainer(
    #     args=training_args,
    #     model=model,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     data_collator=data_collator,
    #     tokenizer=processor.feature_extractor,
    # )
    
    print("Prêt pour l'entraînement local sur CPU.")
    return model, processor

if __name__ == "__main__":
    print("Initialisation du script d'entraînement CPU...")
    # Simulation
    train_whisper_on_cpu(None)
