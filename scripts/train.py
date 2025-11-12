from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import torch

# Load dataset from JSON
data_files = {
    "train": "../data/train.json",
    "validation": "../data/dev.json"
}
dataset = load_dataset("json", data_files=data_files)

# Format input/output for training
def preprocess(example):
    question = example["qa"]["question"]
    answer = example["qa"]["answer"]
    return {"input": f"question: {question}", "target": answer}

dataset = dataset.map(preprocess)

# Load tokenizer and model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenize
def tokenize(batch):
    model_inputs = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=256)
    labels = tokenizer(batch["target"], padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/financial-flan-t5",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs"
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./models/financial-flan-t5")
tokenizer.save_pretrained("./models/financial-flan-t5")