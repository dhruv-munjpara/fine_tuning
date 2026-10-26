"""
Session 3 - Task 4: QLoRA & Memory Footprint Monitoring
Module: FINE-TUNING & MODEL CUSTOMIZATION

Task Requirements:
Use QLoRA via the peft library to fine-tune a small language model (such as distilbert-base-uncased)
on your custom dataset. Monitor your system’s RAM usage during training and take a screenshot
showing the difference compared to standard fine-tuning.
Hint: Use any system monitor tool (like Task Manager or top / psutil) to capture memory usage.
"""

import os
import time
import psutil
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 1. Custom Dataset
TWEET_SAMPLES = [
    {"text": "Loving the new camera update, photos look absolutely stunning!", "label": 1},
    {"text": "Best customer service experience ever, resolved in 5 minutes!", "label": 1},
    {"text": "Super fast delivery and the packaging was completely eco-friendly.", "label": 1},
    {"text": "Incredible performance, smooth 120Hz display is a game changer.", "label": 1},
    {"text": "Really happy with this purchase, exceeded all my expectations!", "label": 1},
    {"text": "Battery drains in 3 hours, completely unusable phone.", "label": 0},
    {"text": "Worst support ever, agent was rude and hung up on me.", "label": 0},
    {"text": "Package arrived late, box was damaged and item was broken.", "label": 0},
    {"text": "App crashes constantly after the latest buggy update, frustrating!", "label": 0},
    {"text": "Total waste of money, do not recommend this product at all.", "label": 0},
]

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_memory_mb(self) -> float:
        """Returns the current process RSS memory in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def get_system_ram_mb(self) -> float:
        """Returns total used system RAM in MB."""
        return psutil.virtual_memory().used / (1024 * 1024)

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

def simulate_standard_finetuning(model_name, dataloader, monitor):
    print("\n" + "=" * 60)
    print("Executing Standard Fine-Tuning (All Weights Trainable)...")
    print("=" * 60)
    mem_before = monitor.get_memory_mb()
    
    # Load full unquantized model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # Ensure all parameters require grad
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    memory_history = []
    model.train()
    for epoch in range(1, 3):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            mem_current = monitor.get_memory_mb()
            memory_history.append(mem_current)
            time.sleep(0.05)

    mem_after = monitor.get_memory_mb()
    peak_mem = max(memory_history) if memory_history else mem_after
    print(f"Standard Fine-Tuning Peak Process RAM: {peak_mem:.2f} MB (Delta: +{peak_mem - mem_before:.2f} MB)")
    
    del model, optimizer
    import gc
    gc.collect()
    return memory_history, peak_mem

def simulate_qlora_finetuning(model_name, dataloader, monitor):
    print("\n" + "=" * 60)
    print("Executing QLoRA / PEFT Fine-Tuning (Quantized/Frozen Base + LoRA)...")
    print("=" * 60)
    mem_before = monitor.get_memory_mb()

    # Configure 4-bit quantization if CUDA available, else load base with frozen quantized emulation
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            num_labels=2
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # On CPU systems, base model is loaded and frozen, simulating QLoRA memory efficiency
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none"
    )

    peft_model = get_peft_model(model, lora_config)
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=5e-4)

    memory_history = []
    peft_model.train()
    for epoch in range(1, 3):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = peft_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            mem_current = monitor.get_memory_mb()
            memory_history.append(mem_current)
            time.sleep(0.05)

    mem_after = monitor.get_memory_mb()
    peak_mem = max(memory_history) if memory_history else mem_after
    print(f"QLoRA / PEFT Fine-Tuning Peak Process RAM: {peak_mem:.2f} MB")
    
    del peft_model, optimizer
    import gc
    gc.collect()
    return memory_history, peak_mem

def run_task_4():
    monitor = MemoryMonitor()
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SimpleDataset(TWEET_SAMPLES, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("=" * 70)
    print("Session 3 - Task 4: Memory Usage Comparison (Standard vs QLoRA/PEFT)")
    print("=" * 70)

    # 1. Standard Fine-Tuning
    std_history, std_peak = simulate_standard_finetuning(model_name, dataloader, monitor)

    # 2. QLoRA / PEFT Fine-Tuning
    qlora_history, qlora_peak = simulate_qlora_finetuning(model_name, dataloader, monitor)

    # 3. Generate Comparison Plot & Save Artifact
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    plot_path = os.path.join(assets_dir, "memory_usage_comparison.png")

    plt.figure(figsize=(10, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Bar chart of peak RAM & Trainable parameters
    categories = ["Standard Full Fine-Tuning", "QLoRA / PEFT Adaptation"]
    peaks = [std_peak, qlora_peak]
    colors = ["#e74c3c", "#2ecc71"]

    plt.bar(categories, peaks, color=colors, width=0.45, edgecolor="black", alpha=0.85)
    plt.title("System RAM Memory Footprint Comparison\nStandard Fine-Tuning vs QLoRA / PEFT", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Peak Process RAM Usage (MB)", fontsize=12)
    plt.ylim(0, max(peaks) * 1.25)

    for i, v in enumerate(peaks):
        plt.text(i, v + (max(peaks) * 0.03), f"{v:.1f} MB", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\n" + "=" * 70)
    print(f"Memory monitoring chart successfully saved to:\n{plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_task_4()
