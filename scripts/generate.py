from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./models/financial-flan-t5"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ask(question):
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🧠 {answer}")

# Example
ask("What is the interest expense in 2009?")