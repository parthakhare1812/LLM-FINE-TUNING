# 🚀 TinyLlama Fine-Tuning with LoRA & QLoRA

> Fine-tuning **TinyLlama-1.1B-Chat** using **QLoRA (4-bit Quantization)** and **LoRA adapters** for efficient instruction tuning on custom datasets.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-green)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![BitsAndBytes](https://img.shields.io/badge/BitsAndBytes-QLoRA-purple)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📖 Overview

This project demonstrates how to efficiently fine-tune **TinyLlama-1.1B-Chat-v1.0** using **QLoRA** with Hugging Face's PEFT ecosystem.

Instead of updating billions of model parameters, the project trains lightweight LoRA adapters on a custom instruction dataset while keeping the base model frozen. This significantly reduces GPU memory requirements while achieving high-quality task-specific performance.

The project includes:

* 4-bit model quantization
* LoRA-based parameter-efficient fine-tuning
* Supervised Fine-Tuning (SFT)
* Automatic adapter saving
* Adapter merging for deployment
* Custom instruction dataset support
* Fast inference after training

---

# ✨ Features

* ✅ TinyLlama 1.1B Chat Model
* ✅ QLoRA (4-bit Quantization)
* ✅ LoRA Adapter Training
* ✅ Hugging Face Transformers
* ✅ TRL SFTTrainer
* ✅ PEFT Integration
* ✅ BitsAndBytes Optimization
* ✅ GPU Accelerated Training
* ✅ Adapter Merging
* ✅ Ready for Deployment

---

# 🏗 Project Structure

```
LLM-Fine-Tuning/
│
├── data/
│   └── dataset.jsonl
│
├── output/
│   └── tinyllama-finetuned-v1/
│
├── train.py
├── inference.py
├── requirements.txt
├── README.md
│
└── assets/
    └── architecture.png
```

---

# ⚙️ Tech Stack

| Category     | Technologies             |
| ------------ | ------------------------ |
| Language     | Python                   |
| Framework    | PyTorch                  |
| Model        | TinyLlama-1.1B-Chat-v1.0 |
| Training     | TRL                      |
| Fine-Tuning  | LoRA                     |
| Quantization | QLoRA (4-bit)            |
| Tokenizer    | Hugging Face             |
| Dataset      | JSONL                    |
| Optimization | BitsAndBytes             |

---

# 🧠 Model Architecture

```
                  Custom Dataset
                        │
                        ▼
              Prompt Formatting
                        │
                        ▼
           TinyLlama Tokenizer
                        │
                        ▼
      TinyLlama Base Model (Frozen)
                        │
             4-bit Quantization
                        │
                        ▼
               LoRA Adapters
                        │
                        ▼
             Supervised Fine-Tuning
                        │
                        ▼
          Trained Adapter Weights
                        │
                        ▼
          Merge with Base Model
                        │
                        ▼
          Optimized Inference Model
```

---

# 📂 Dataset Format

The dataset should be in **JSONL** format.

Example:

```json
{
    "instruction": "Explain gravity.",
    "input": "",
    "output": "Gravity is the force that attracts objects toward each other."
}
```

Each record must contain:

* instruction
* input
* output

---

# 📥 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/LLM-Fine-Tuning.git
```

Navigate to the project

```bash
cd LLM-Fine-Tuning
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Required Packages

```text
torch
transformers
datasets
trl
peft
accelerate
bitsandbytes
sentencepiece
```

Install manually

```bash
pip install torch transformers datasets trl peft accelerate bitsandbytes sentencepiece
```

---

# 🚀 Training

Run

```bash
python train.py
```

Training pipeline

```
Load TinyLlama

↓

Load Dataset

↓

Quantize Model (4-bit)

↓

Prepare LoRA

↓

Apply PEFT

↓

Train using SFTTrainer

↓

Save Adapter

↓

Save Tokenizer
```

---

# ⚙️ Training Configuration

| Parameter             | Value                    |
| --------------------- | ------------------------ |
| Base Model            | TinyLlama-1.1B-Chat-v1.0 |
| Quantization          | 4-bit                    |
| LoRA Rank             | 16                       |
| Alpha                 | 16                       |
| Dropout               | 0.05                     |
| Learning Rate         | 2e-4                     |
| Batch Size            | 4                        |
| Gradient Accumulation | 4                        |
| Max Sequence Length   | 512                      |
| Max Steps             | 100                      |
| Precision             | FP16                     |

---

# 🧩 LoRA Target Modules

The project fine-tunes the following transformer layers:

```
q_proj

k_proj

v_proj

o_proj
```

Only these layers are updated, reducing the number of trainable parameters dramatically.

---

# 💾 Saved Model

After training

```
output/

└── tinyllama-finetuned-v1/

      adapter_model.safetensors

      adapter_config.json

      tokenizer.json

      tokenizer_config.json

      special_tokens_map.json
```

---

# ⚡ Inference

Run

```bash
python inference.py
```

Example prompt

```
### Instruction:
Explain the concept of gravity.

### Input:

### Response:
```

Expected workflow

```
Load Base Model

↓

Load LoRA Adapter

↓

Merge Adapter

↓

Generate Response

↓

Decode Output
```

---

# 📈 Performance Benefits

Compared to Full Fine-Tuning

✅ Much lower GPU memory

✅ Faster training

✅ Lower storage requirements

✅ Better scalability

✅ Easy adapter sharing

---

# 🎯 Applications

This project can be adapted for:

* Customer Support Bots
* AI Tutors
* Healthcare Assistants
* Coding Assistants
* Legal Document Analysis
* Educational Chatbots
* Enterprise Knowledge Systems
* Research Assistants
* FAQ Automation
* Domain-Specific LLMs

---

# 🔮 Future Improvements

* Multi-GPU Training
* DeepSpeed Integration
* Flash Attention
* Evaluation Pipeline
* Hugging Face Hub Integration
* Weights & Biases Logging
* Gradio Demo
* Streamlit UI
* RAG Integration
* Model Quantization Benchmarks
* Docker Support
* Kubernetes Deployment

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📚 References

* TinyLlama
* Hugging Face Transformers
* Hugging Face PEFT
* Hugging Face TRL
* BitsAndBytes
* QLoRA Paper
* LoRA Paper

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project helpful:

⭐ Star the repository

🍴 Fork it

🐛 Report issues

💡 Suggest improvements

---

## 👨‍💻 Author

**Partha Khare**

AI Engineer | Machine Learning Enthusiast | LLM Engineer | Computer Vision | Generative AI | Open Source

If you enjoyed this project, consider giving it a ⭐ and connecting with me on GitHub and LinkedIn.
