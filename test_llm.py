from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/share/pi/nigam/users/calebwin/hf_cache/ckpts/med-s1-1k-tuned"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu") 
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer("Hi there. How are you?", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits  # shape [batch_size, seq_len, vocab_size]
    print("Any NaNs in logits? ", torch.isnan(logits).any())
