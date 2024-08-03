!pip install -U "transformers[sentencepiece]" "accelerate" "safetensors" "torch"
!pip install -U "optimum==1.12.0" "auto-gptq==0.4.2"

hugging_face_dataset_id = "wikitext2"
from optimum.gptq import GPTQQuantizer

gptq_quantizer = GPTQQuantizer(bits=4,
                               dataset = hugging_face_dataset_id,
                               model_seqlen=4096)
gptq_quantizer.quant_method = "gptq"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             low_cpu_mem_usage=True,
                                             torch_dtype=torch.float16)
import os
import json

save_folder = "/content/phi-2_quantised"

quantized_model = gptq_quantizer.quantize_model(model, tokenizer)

quantized_model.save_pretrained(save_folder, safe_serialization=True)

fast_tokenizer = AutoTokenizer.from_pretrained(model_id)
fast_tokenizer.save_pretrained(save_folder)

with open(os.path.join(save_folder, "quantize_config.json"), "w", encoding="utf-8") as config_file:
    gptq_quantizer.disable_exllama = False
    json.dump(gptq_quantizer.to_dict(), config_file, indent=2)

import shutil
import os

source_folder = '/content/phi-2_quantised/model.safetensors'
destination_folder = '/content/drive/My Drive/Phi-2-quant/'

os.makedirs(destination_folder, exist_ok=True)

shutil.copy(source_folder, destination_folder)
