import os
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# === Configurations ===
DATA_FILE = "data/mydata.txt"   # apna dataset file
MODEL_NAME = "gpt2"             # "gpt2-medium" bhi try kar sakte ho agar GPU hai
OUTPUT_DIR = "finetuned_model"
BLOCK_SIZE = 128
EPOCHS = 3
LR = 2e-5
BATCH_SIZE = 2

# === Load dataset ===
assert os.path.exists(DATA_FILE), f"Dataset not found at {DATA_FILE}"
print(f"📂 Loading dataset from {DATA_FILE} ...")
raw_ds = load_dataset("text", data_files={"train": DATA_FILE})

# === Tokenizer ===
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized = raw_ds.map(tokenize_function, batched=True, remove_columns=["text"])

# === Group texts into blocks ===
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_ds = tokenized.map(group_texts, batched=True)

# === Model ===
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.eos_token_id

# === Data Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Training Args ===
training_args = TrainingArguments(
    output_dir="results",
    overwrite_output_dir=True,
    evaluation_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    logging_steps=50,
    logging_dir="logs",
    save_steps=500,
    save_total_limit=2,
    fp16=False,     # ✅ force disable mixed precision (works on CPU)
    report_to=["none"],
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_ds["train"],
    data_collator=data_collator,
)

print("🚀 Starting training...")
trainer.train()

# === Save model ===
print(f"💾 Saving model to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete! Model saved at:", OUTPUT_DIR)
