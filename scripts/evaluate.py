from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
from tqdm import tqdm

# Load model
model_path = "./models/financial-flan-t5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.eval()

# Load dataset
dataset = load_dataset("json", data_files={"validation": "./data/dev.json"})["validation"]

# Preprocess for T5-style input
def preprocess(example):
    return {
        "input": f"question: {example['qa']['question']}",
        "target": example["qa"]["answer"]
    }

dataset = dataset.map(preprocess)

# Metrics
rouge = load_metric("rouge")
bleu = load_metric("bleu")

generated_answers = []
reference_answers = []

# Inference loop
for sample in tqdm(dataset, desc="Evaluating"):
    inputs = tokenizer(sample["input"], return_tensors="pt", truncation=True, padding=True).to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reference = sample["target"]

    generated_answers.append(prediction)
    reference_answers.append(reference)

# Metric calculations
rouge_output = rouge.compute(predictions=generated_answers, references=reference_answers)
bleu_input = [[ref.split()] for ref in reference_answers]
bleu_output = [pred.split() for pred in generated_answers]
bleu_score = bleu.compute(predictions=bleu_output, references=bleu_input)

# Display evaluation summary
print("\n📊 Evaluation Results:")
print(f"ROUGE-L: {rouge_output['rougeL'].mid.fmeasure:.4f}")
print(f"BLEU: {bleu_score['bleu']:.4f}")

# Sample outputs
print("\n📌 Sample Predictions:")
for i in range(3):
    print(f"\nQ: {dataset[i]['input']}")
    print(f"A (True): {reference_answers[i]}")
    print(f"A (Pred): {generated_answers[i]}")
