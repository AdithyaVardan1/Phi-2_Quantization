!pip -q install auto-gptq
!pip -q install optimum
!pip -q install bitsandbytes
!pip -q install einops

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model.eval()

comment = "Great content, thank you!"
prompt=f'''Instruct : {comment} \n Output : '''

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=200)

print(tokenizer.batch_decode(outputs)[0])
intstructions_string = f"""AVMGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature 'AVMGPT'. \
AVMGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''Instruct : {intstructions_string} \n{comment} \n Output : '''

prompt = prompt_template(comment)
print(prompt)
# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])
model.train()
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()
# load dataset
data = load_dataset("shawhin/shawgpt-youtube-comments")
# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

tokenizer.pad_token = tokenizer.eos_token
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "avmgpt-ft",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=False,
    optim="adamw_8bit",

)

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)

# train model
model.config.use_cache = False
trainer.train()

# renable warnings
model.config.use_cache = True
from huggingface_hub import notebook_login
notebook_login()

model_id = 'Phi2quant' + "/" + "AVMGPT"
print(model_id)
token1 = "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"
model.push_to_hub(model_id, token = token1)
trainer.push_to_hub(model_id)

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM,AutoTokenizer

model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("Phi2quant/AVMGPT")
model = PeftModel.from_pretrained(model, "Phi2quant/AVMGPT")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
intstructions_string = f"""AVMGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–AVMGPT. \
AVMGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""
prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

comment = "Great content, thank you!"

prompt = prompt_template(comment)
print(prompt)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])
comment = "What is fat-tailedness?"
prompt = prompt_template(comment)

model.eval()
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
print(tokenizer.batch_decode(outputs)[0])