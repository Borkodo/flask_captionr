import sqlite3
import pandas as pd
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

print("[Step 1] Connecting to SQLite database...")

conn = sqlite3.connect('instagram_captions.db')
print("  > Connected successfully to SQLite database.")

print("[Step 2] Reading data from database...")

query = "SELECT * FROM CaptionsTable"

df = pd.read_sql_query(query, conn)

conn.close()
print(f"  > Extracted {len(df)} rows from database.")

print("[Step 3] Extracting captions from the dataset...")

captions = df['Captions'].tolist()
print(f"  > Extracted {len(captions)} captions.")

print("[Step 4] Initializing the tokenizer and model...")


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
print("  > Tokenizer and model initialized.")

print("[Step 5] Preparing and tokenizing the dataset...")

with open("captions_dataset.txt", "w") as f:
    f.write("\n".join(captions))
print("  > Dataset prepared.")


dataset = load_dataset('text', data_files={'train': 'captions_dataset.txt'})
print("  > Dataset loaded.")



def tokenize_function(examples):
    return tokenizer(examples['text'])


tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("  > Dataset tokenized.")

print("[Step 6] Initializing training arguments...")
# Initialize training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
)
print("  > Training arguments initialized.")

print("[Step 7] Initializing the Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    ),
    train_dataset=tokenized_dataset['train'],
)
print("  > Trainer initialized.")

print("[Step 8] Fine-tuning the model...")

trainer.train()
print("  > Model fine-tuned.")

print("[Step 9] Saving the fine-tuned model...")

model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("  > Fine-tuned model saved.")

print("[Step 10] All steps completed.")
