import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig  # UPDATED: Import SFTConfig

# --- Configuration ---
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
NEW_MODEL_NAME = "tinyllama-finetuned-v1"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "output"

# --- 1. Load and Quantize Model (4-bit) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# --- 2. Tokenizer Setup ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"          

# --- 3. Prepare Model for LoRA (Adapters) ---
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=16,                   
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() 

# --- 4. Load Dataset ---
def formatting_prompts_func(examples):
    output_texts = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        output_texts.append(text)
    return output_texts

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# --- 5. Training Arguments (UPDATED) ---
# SFTConfig replaces TrainingArguments for SFTTrainer specific args
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_seq_length=512,          # MOVED: Now part of SFTConfig
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=100,
    warmup_ratio=0.03,
    save_steps=50,
    group_by_length=True,
    # dataset_text_field="text", # Optional: uncomment if you stop using formatting_func
)

# --- 6. Initialize Trainer (UPDATED) ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    args=training_args,          # Pass the SFTConfig here
)

# --- 7. Train and Save ---
print("Starting training...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}/{NEW_MODEL_NAME}")
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, NEW_MODEL_NAME))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, NEW_MODEL_NAME))