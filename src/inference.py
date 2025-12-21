import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "output/tinyllama-finetuned-v1"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload() # Merges weights for faster inference

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Test the model
prompt = "### Instruction:\nExplain the concept of gravity.\n\n### Input:\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating response...")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))