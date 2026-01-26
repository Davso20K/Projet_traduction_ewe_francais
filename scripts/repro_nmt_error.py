from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset

model_name = "facebook/nllb-200-distilled-600M"
# Fix: Initialize with src_lang and tgt_lang
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="ewe_Latn", tgt_lang="ewe_Latn")

# Mimic the dataset
data = {
    "mina": ["So gɔ̃mèjèje a, Mawu ɖo jiŋùkusi ku ànyigbã.", "Ànyigbã lè nyàmà, gbà lè gbalo"],
    "ewe": ["Le gɔmedzedzea me, Mawu wɔ dziƒo kple anyigba.", "Anyigba nɔ nyama, nɔnɔme menɔe nɛ o"]
}
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

def preprocess_function(examples):
    print(f"DEBUG: Processing batch of size {len(examples['mina'])}")
    try:
        # No need for text_target if we don't use the labels here, 
        # but let's see why it failed.
        model_inputs = tokenizer(
            examples["mina"], 
            text_target=examples["ewe"], 
            max_length=128, 
            truncation=True
        )
        print("DEBUG: Success")
        return model_inputs
    except Exception as e:
        print(f"DEBUG: Error -> {e}")
        import traceback
        traceback.print_exc()
        raise e

print("Running map...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
print("Done.")
