"""
Session 3 - Task 1: LoRA Parameter Count Comparison
Module: FINE-TUNING & MODEL CUSTOMIZATION

Task Requirements:
Install the peft library and load a pre-trained sentiment analysis model
using Hugging Face Transformers. Print the model’s total number of trainable
parameters before and after applying LoRA using peft’s LoraConfig.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

def compare_lora_parameters(model_name: str = "distilbert-base-uncased", num_labels: int = 2):
    print("=" * 70)
    print("Session 3 - Task 1: Trainable Parameter Comparison with LoRA")
    print("=" * 70)
    print(f"Loading Base Model: {model_name} ...")

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # 1. Total trainable parameters before LoRA
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n[1] Before Applying LoRA (Standard Pre-Trained Model):")
    print("-" * 50)
    print(f"Total Parameters     : {total_params_before:,}")
    print(f"Trainable Parameters : {trainable_params_before:,} ({trainable_params_before / total_params_before * 100:.2f}%)")
    print(f"Frozen Parameters    : {total_params_before - trainable_params_before:,}")

    # 2. Configure LoRA via PEFT
    # Target modules for DistilBERT attention are q_lin and v_lin
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                  # Rank
        lora_alpha=16,        # Alpha scaling
        target_modules=["q_lin", "v_lin"], # Attention projection matrices
        lora_dropout=0.1,
        bias="none"
    )

    print("\n[2] Applying LoRA Configuration:")
    print("-" * 50)
    print(f"Task Type      : {lora_config.task_type}")
    print(f"LoRA Rank (r)  : {lora_config.r}")
    print(f"LoRA Alpha     : {lora_config.lora_alpha}")
    print(f"Target Modules : {lora_config.target_modules}")

    # 3. Wrap model with PEFT LoRA
    peft_model = get_peft_model(model, lora_config)

    # 4. Parameters After Applying LoRA
    print("\n[3] After Applying LoRA (PEFT Model):")
    print("-" * 50)
    peft_model.print_trainable_parameters()

    total_params_after = sum(p.numel() for p in peft_model.parameters())
    trainable_params_after = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    frozen_params_after = total_params_after - trainable_params_after

    print(f"\nDetailed Breakdown:")
    print(f"Total Parameters     : {total_params_after:,}")
    print(f"Trainable Parameters : {trainable_params_after:,} ({trainable_params_after / total_params_after * 100:.4f}%)")
    print(f"Frozen Parameters    : {frozen_params_after:,} ({frozen_params_after / total_params_after * 100:.2f}%)")
    print(f"Parameter Reduction  : {(1 - (trainable_params_after / trainable_params_before)) * 100:.2f}% fewer trainable weights!")

    print("\n" + "=" * 70)
    print("Task 1 Completed Successfully!")
    print("=" * 70)
    return peft_model

if __name__ == "__main__":
    compare_lora_parameters()
