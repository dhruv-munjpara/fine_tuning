"""
Session 1 - Task 4: Inspect Architecture and Freeze Layers
Module: FINE-TUNING & MODEL CUSTOMIZATION

Task Requirements:
Write a Python script using Hugging Face Transformers to load a pre-trained model
(like BERT or DistilBERT), inspect its architecture, count total trainable parameters,
and freeze the base layers.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def inspect_and_freeze(model_name: str = "distilbert-base-uncased", num_labels: int = 2):
    print("=" * 70)
    print(f"Loading Pre-Trained Model: {model_name}")
    print("=" * 70)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # 1. Inspect Architecture
    print("\n[1] Model Architecture Overview:")
    print("-" * 50)
    print(model)
    
    # Helper to calculate parameters
    def get_parameter_counts(m):
        total_p = sum(p.numel() for p in m.parameters())
        trainable_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        frozen_p = total_p - trainable_p
        return total_p, trainable_p, frozen_p

    # 2. Count Initial Parameters
    total_params, trainable_params, frozen_params = get_parameter_counts(model)
    print("\n[2] Initial Parameter Statistics (Before Freezing):")
    print("-" * 50)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"Frozen Parameters    : {frozen_params:,}")

    # 3. Freeze Base Transformer Layers
    # In DistilBERT, base model is accessible via model.distilbert
    # In BERT, it's accessible via model.bert
    base_model_attr = "distilbert" if hasattr(model, "distilbert") else ("bert" if hasattr(model, "bert") else None)
    
    if base_model_attr:
        base_model = getattr(model, base_model_attr)
        for param in base_model.parameters():
            param.requires_grad = False
        print(f"\n[3] Successfully froze base layers ({base_model_attr})!")
    else:
        # Fallback: freeze everything except classification head (pre_classifier & classifier)
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("\n[3] Successfully froze base layers (classifier left trainable)!")

    # 4. Count Parameters After Freezing
    total_params_after, trainable_params_after, frozen_params_after = get_parameter_counts(model)
    print("\n[4] Parameter Statistics After Freezing Base Layers:")
    print("-" * 50)
    print(f"Total Parameters     : {total_params_after:,}")
    print(f"Trainable Parameters : {trainable_params_after:,} ({trainable_params_after / total_params_after * 100:.2f}%)")
    print(f"Frozen Parameters    : {frozen_params_after:,}")
    
    # 5. List which layers remain trainable
    print("\n[5] Trainable Layers Summary:")
    print("-" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name:40s} | Shape: {tuple(param.shape)} | Elements: {param.numel():,}")

    print("\n" + "=" * 70)
    print("Task 4 Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    inspect_and_freeze()
