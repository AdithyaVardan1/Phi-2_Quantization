import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tuning_device = "cuda:0"  
dtype = "auto" if tuning_device != "hpu" else torch.bfloat16
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

from auto_round import AutoRound

bits, group_size, sym = 4, 128, False
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, device=tuning_device)
autoround.quantize()
output_dir = "./tmp_autoround"
autoround.save_quantized(output_dir)
