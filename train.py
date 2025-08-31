import os
from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# === Step 1: Configuration ===
def get_config():
    DATA_FILE = "data/mydata.txt"
    MODEL_NAME = "gpt2"
    OUTPUT_DIR = "finetuned_model"
    BLOCK_SIZE = 128
    EPOCHS = 3
    LR = 2e-5
    BATCH_SIZE = 2
    return DATA_FILE, MODEL_NAME, OUTPUT_DIR, BLOCK_SIZE, EPOCHS, LR, BATCH_SIZE

# === Step 2: Check dataset file exists ===
def check_dataset_file(path):
    print(f"Checking if dataset file exists at {path} ...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at {path}")
    print("Dataset file found.")

# === Step 3: Load dataset ===
def load_text_dataset(data_file):
    print(f"Loading dataset from {data_file} ...")
    dataset = load_dataset("text", data_files={"train": data_file})
    print(f"Dataset loaded with {len(dataset['train'])} samples.")
    return dataset

# === Step 4: Load tokenizer ===
def load_tokenizer(model_name):
    print(f"Loading tokenizer for {model_name} ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    print(f"Original pad token: {tokenizer.pad_token}")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to EOS token: {tokenizer.pad_token}")
    return tokenizer

# === Step 5: Tokenize dataset ===
def tokenize_dataset(dataset, tokenizer):
    print("Tokenizing dataset ...")
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print("Tokenization complete.")
    return tokenized

# === Step 6: Group texts into blocks ===
def group_texts(tokenized_dataset, block_size):
    print(f"Grouping texts into blocks of size {block_size} ...")
    def group_function(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    grouped = tokenized_dataset.map(group_function, batched=True)
    print(f"Grouping complete. Number of samples: {len(grouped['train'])}")
    return grouped

# === Step 7: Load model ===
def load_model(model_name, tokenizer):
    print(f"Loading pre-trained model {model_name} ...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    print(f"Model pad_token_id set to {model.config.pad_token_id}")
    return model

# === Step 8: Prepare data collator ===
def prepare_data_collator(tokenizer):
    print("Preparing data collator ...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return data_collator

# === Step 9: Setup training arguments ===
def setup_training_args(batch_size, epochs, lr):
    print("Setting up training arguments ...")
    training_args = TrainingArguments(
        output_dir="results",
        overwrite_output_dir=True,
        evaluation_strategy="no",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        logging_dir="logs",
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        report_to=["none"],
    )
    print("Training arguments ready.")
    return training_args

# === Step 10: Initialize Trainer ===
def initialize_trainer(model, training_args, train_dataset, data_collator):
    print("Initializing Trainer ...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    print("Trainer initialized.")
    return trainer

# === Step 11: Train model ===
def train_model(trainer):
    print("Starting training ...")
    trainer.train()
    print("Training finished.")

# === Step 12: Save model and tokenizer ===
def save_model_and_tokenizer(trainer, tokenizer, output_dir):
    print(f"Saving model and tokenizer to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully.")

# === Main function ===
def main():
    DATA_FILE, MODEL_NAME, OUTPUT_DIR, BLOCK_SIZE, EPOCHS, LR, BATCH_SIZE = get_config()
    check_dataset_file(DATA_FILE)
    dataset = load_text_dataset(DATA_FILE)
    tokenizer = load_tokenizer(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    grouped_dataset = group_texts(tokenized_dataset, BLOCK_SIZE)
    model = load_model(MODEL_NAME, tokenizer)
    data_collator = prepare_data_collator(tokenizer)
    training_args = setup_training_args(BATCH_SIZE, EPOCHS, LR)
    trainer = initialize_trainer(model, training_args, grouped_dataset["train"], data_collator)
    train_model(trainer)
    save_model_and_tokenizer(trainer, tokenizer, OUTPUT_DIR)
    print("✅ All done!")

if __name__ == "__main__":
    main()