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
DATA_FILE = "data/mydata.txt"   # Apna dataset file ka path
MODEL_NAME = "gpt2"             # "gpt2-medium" bhi try kar sakte ho agar GPU available hai
OUTPUT_DIR = "finetuned_model"  # Jahan model save karna hai
BLOCK_SIZE = 128                # Sequence length for training
EPOCHS = 3                     # Kitne epochs tak train karna hai
LR = 2e-5                      # Learning rate
BATCH_SIZE = 2                 # Batch size

# === Step 1: Check if dataset file exists ===
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}")
print(f"📂 Loading dataset from {DATA_FILE} ...")

# === Step 2: Load dataset using Huggingface datasets ===
raw_ds = load_dataset("text", data_files={"train": DATA_FILE})
print(f"Dataset loaded. Number of samples: {len(raw_ds['train'])}")

# === Step 3: Load tokenizer ===
print(f"Loading tokenizer for model {MODEL_NAME} ...")
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

# GPT2 tokenizer does not have a pad token by default, so set it to eos token
tokenizer.pad_token = tokenizer.eos_token

# === Step 4: Tokenize the dataset ===
def tokenize_function(examples):
    # Tokenize the text column
    return tokenizer(examples["text"])

print("Tokenizing dataset ...")
tokenized_ds = raw_ds.map(tokenize_function, batched=True, remove_columns=["text"])
print("Tokenization complete.")

# === Step 5: Group texts into blocks of BLOCK_SIZE ===
def group_texts(examples):
    # Concatenate all input_ids into a single list
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    # Calculate total length divisible by BLOCK_SIZE
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    # Split into chunks of BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    # Labels are same as input_ids for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

print(f"Grouping texts into blocks of size {BLOCK_SIZE} ...")
lm_ds = tokenized_ds.map(group_texts, batched=True)
print(f"Number of training samples after grouping: {len(lm_ds['train'])}")

# === Step 6: Load pre-trained GPT2 model ===
print(f"Loading pre-trained model {MODEL_NAME} ...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# Set pad_token_id to eos_token_id to avoid warnings during training
model.config.pad_token_id = tokenizer.eos_token_id

# === Step 7: Prepare data collator for language modeling ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Step 8: Define training arguments ===
training_args = TrainingArguments(
    output_dir="results",            # Directory to save checkpoints and logs
    overwrite_output_dir=True,       # Overwrite existing output directory
    evaluation_strategy="no",        # No evaluation during training
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.01,
    logging_steps=50,                # Log every 50 steps
    logging_dir="logs",              # Directory for logs
    save_steps=500,                 # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only last 2 checkpoints
    fp16=False,                     # Disable mixed precision (useful if no GPU)
    report_to=["none"],             # Disable reporting to external services
)

# === Step 9: Initialize Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_ds["train"],
    data_collator=data_collator,
)

# === Step 10: Start training ===
print("🚀 Starting training...")
trainer.train()

# === Step 11: Save the fine-tuned model and tokenizer ===
print(f"💾 Saving model and tokenizer to {OUTPUT_DIR} ...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete! Model saved at:", OUTPUT_DIR)