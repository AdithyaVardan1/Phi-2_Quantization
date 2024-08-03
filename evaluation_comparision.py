!pip install -U "transformers[sentencepiece]" "accelerate" "safetensors" "torch" -q
!pip install -U "optimum==1.12.0" "auto-gptq==0.4.2" -q
!pip install einops -q
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil
from nltk.translate.bleu_score import sentence_bleu
import torch
torch.set_default_device("cuda")
quantized_model_dir = "AVMLegend/Phi-2" #or microsoft/phi-2 or Intel/phi-2-int4-inc or TheBloke/phi-2-#GPTQ


model = AutoModelForCausalLM.from_pretrained(quantized_model_dir, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

prompt = "Explain logistic regression."

start_time = time.time()

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_new_tokens=50)
first_token_generation_time = time.time() - start_time
text = tokenizer.batch_decode(outputs)[0]

last_stop_index = text.rfind(".")
if last_stop_index != -1:
    text = text[:last_stop_index + 1]



reference_text = "Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable."
bleu_score = sentence_bleu([reference_text.split()], text.split())

with torch.no_grad():
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = tokenizer.encode(text, return_tensors="pt")
    logits = model(input_ids=input_ids.to(model.device)).logits

latency = time.time() - start_time

print("Generated Text:")
print(text)
print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("Total Latency (seconds):", latency)
print("Time taken to generate first token (seconds):", first_token_generation_time)
print("BLEU Score:", bleu_score)
