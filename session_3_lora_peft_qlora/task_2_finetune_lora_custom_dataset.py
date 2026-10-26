"""
Session 3 - Task 2: Fine-Tune Text Classification with LoRA on Custom Dataset
Module: FINE-TUNING & MODEL CUSTOMIZATION

Task Requirements:
Fine-tune a pre-trained text classification model on a small custom dataset
(for example, 10 sample tweets labeled as positive or negative) using LoRA with peft.
Save the adapted model weights to disk and report the file size.
Hint: Use peft's get_peft_model and LoraConfig for efficient fine-tuning.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# 1. Custom Dataset: 10 sample tweets labeled as Positive (1) or Negative (0)
CUSTOM_TWEET_DATASET = [
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

class TweetDataset(Dataset):
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

def train_and_save_lora():
    model_name = "distilbert-base-uncased"
    output_dir = os.path.join(os.path.dirname(__file__), "saved_lora_adapter")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Session 3 - Task 2: LoRA Fine-Tuning & Adapter Weight Saving")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Load Tokenizer & Base Model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.1,
        bias="none"
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)
    print("\nTrainable parameters under LoRA:")
    peft_model.print_trainable_parameters()

    # Prepare DataLoader
    dataset = TweetDataset(CUSTOM_TWEET_DATASET, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=5e-4)
    epochs = 4

    print(f"\nTraining LoRA Adapter for {epochs} epochs on 10 custom tweet samples...")
    peft_model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} | Average Training Loss: {avg_loss:.4f}")

    # 4. Save adapted model weights to disk
    print("\n" + "=" * 50)
    print(f"Saving adapted LoRA weights to: {output_dir}")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("LoRA adapter weights and config successfully saved!")

    # 5. Inspect and report file sizes
    print("\n[Disk Footprint Report]:")
    print("-" * 50)
    total_size_bytes = 0
    for root, _, files in os.walk(output_dir):
        for f in files:
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            total_size_bytes += size
            print(f" - {f:30s} : {size / 1024:.2f} KB ({size:,} bytes)")

    print("-" * 50)
    print(f"Total Adapter Size on Disk: {total_size_bytes / (1024 * 1024):.2f} MB ({total_size_bytes / 1024:.2f} KB)")
    print("Notice: A full model checkpoint would take ~268 MB, whereas this LoRA adapter is ~2.9 MB!")
    print("=" * 70)

if __name__ == "__main__":
    train_and_save_lora()
