import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
from tqdm import tqdm
import torch
import os

# Load data
df = pd.read_csv(os.path.join("..", "data", "enhanced_ai_llm_dataset.csv"))

# Train-test split
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

# Rename columns to match model expectations
dataset = dataset.rename_column("text", "document")
dataset = dataset.rename_column("summary", "summary")

# Load tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Dynamic max length
tokenized_texts = tokenizer(dataset['train']['document'], truncation=False)
tokenized_summaries = tokenizer(dataset['train']['summary'], truncation=False)
max_input_length = int(np.percentile([len(x) for x in tokenized_texts['input_ids']], 75))
max_summary_length = int(np.percentile([len(x) for x in tokenized_summaries['input_ids']], 75))

# Tokenization
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["document"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary"],
            max_length=max_summary_length,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training setup
training_args = Seq2SeqTrainingArguments(
    output_dir="../results",
    eval_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    logging_dir='../logs',
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

# Evaluation
rouge = evaluate.load("rouge")

def generate_summaries(texts):
    generated = []
    for text in tqdm(texts, desc="Generating summaries"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_input_length).to(model.device)
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_summary_length,
            num_beams=4,
            early_stopping=True
        )
        generated.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return generated

predictions = generate_summaries(dataset["test"]["document"])
references = dataset["test"]["summary"]
results = rouge.compute(predictions=predictions, references=references)

print("\n📈 ROUGE scores after fine-tuning:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# Save model
model.save_pretrained("../fine_tuned_bart_model")
tokenizer.save_pretrained("../fine_tuned_bart_model")
