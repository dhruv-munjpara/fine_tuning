# Session 3: LoRA, PEFT & QLoRA Deep Dive
**Module:** Fine-Tuning & Model Customization  
**Student Repository:** [https://github.com/dhruv-munjpara/fine_tuning](https://github.com/dhruv-munjpara/fine_tuning)

---

## 📌 Task 1: Trainable Parameters Before and After LoRA

The complete Python script is located in [`task_1_lora_trainable_params.py`](file:///d:/DA%26DS/fine_tuning/session_3_lora_peft_qlora/task_1_lora_trainable_params.py).

### Results Summary:
* **Base Model:** `distilbert-base-uncased` with 2 classification labels.
* **LoRA Hyperparameters:**
  - `task_type`: `TaskType.SEQ_CLS`
  - `r (rank)`: `8`
  - `lora_alpha`: `16`
  - `target_modules`: `["q_lin", "v_lin"]`
  - `lora_dropout`: `0.1`

### Exact Parameter Comparison:

| Parameter Category | Before LoRA (Standard Model) | After LoRA (PEFT Adapter) | Impact / Reduction |
| :--- | :--- | :--- | :--- |
| **Total Parameters** | 66,955,010 | 67,694,596 | Base weights + low-rank matrices |
| **Trainable Parameters** | 66,955,010 (100.00%) | **739,586 (1.0925%)** | **98.90% reduction** in trainable parameters! |
| **Frozen Parameters** | 0 | 66,955,010 (98.91%) | Base weights remain untouched |

---

## 📌 Task 2: Fine-Tuning on Custom Tweet Dataset & Adapter Disk Footprint

The implementation is located in [`task_2_finetune_lora_custom_dataset.py`](file:///d:/DA%26DS/fine_tuning/session_3_lora_peft_qlora/task_2_finetune_lora_custom_dataset.py).

### Training Execution:
* **Dataset:** 10 curated user sentiment samples (5 positive product reviews, 5 negative customer complaints).
* **Batch Size:** 2 | **Optimizer:** AdamW (`lr=5e-4`) | **Epochs:** 4
* **Loss Progression:**
  - Epoch 1: 0.8793
  - Epoch 2: 0.5978
  - Epoch 3: 0.4437
  - Epoch 4: **0.3158**

### Saved Adapter Artifacts & Disk Footprint:
The adapted weights and configuration are saved under [`saved_lora_adapter/`](file:///d:/DA%26DS/fine_tuning/session_3_lora_peft_qlora/saved_lora_adapter):

| File Name | Size (KB) | Purpose |
| :--- | :--- | :--- |
| `adapter_model.safetensors` | **2,892.77 KB (~2.89 MB)** | LoRA low-rank weight matrices ($A$ and $B$) and classification head |
| `adapter_config.json` | **1.15 KB** | LoRA hyperparameters, target modules, scaling factors |
| `tokenizer.json` | 694.98 KB | Vocabulary and subword tokenization rules |
| `tokenizer_config.json` | 0.36 KB | Tokenizer settings and special token mappings |
| `README.md` | 5.05 KB | Auto-generated PEFT model card metadata |
| **Total LoRA Adapter Footprint** | **~3.51 MB** | **98.7% smaller** than full model checkpoint (~268 MB) |

---

## 📌 Task 3: Main Benefits of PEFT & Real-World Application Scenarios

### 1. Main Benefits of PEFT over Full Fine-Tuning:
1. **Drastic Storage Efficiency (Multi-Tenancy):** Instead of saving a 14 GB model checkpoint for every individual client or task, we keep a single base model in memory and swap tiny LoRA adapter weights (5 MB – 50 MB) on the fly.
2. **Mitigation of Catastrophic Forgetting:** Full fine-tuning alters all weights across the network, degrading generalized capabilities and common-sense reasoning. PEFT leaves base weights frozen, preserving base foundation capabilities while specializing adapters.
3. **Low Hardware Barrier:** Because optimizer states (e.g. AdamW requires 8 bytes per trainable parameter) are only calculated for <1% of the weights, training memory overhead is reduced by up to 70–80%, enabling fine-tuning on consumer-grade hardware.

### 2. Real-World Application Scenarios:

#### Scenario A: WhatsApp – Smart Replies & On-Device Auto-Summarization
* **Why PEFT is ideal:** WhatsApp operates on hundreds of millions of mobile smartphones with strict storage and battery limits. Storing multiple large fine-tuned models on mobile flash memory is impossible. With PEFT, WhatsApp can deploy a single on-device compressed foundation model and swap micro-LoRA adapters (e.g., 2 MB each for Hindi transliteration, formal business messaging, or group chat summary) directly inside the mobile app runtime.

#### Scenario B: Instagram – Creator Tone Matching & Direct Message Automation
* **Why PEFT is ideal:** High-profile Instagram creators and influencers receive tens of thousands of DMs daily and want AI auto-replies that mirror their unique personal voice, slang, and emojis. Meta cannot afford to host and serve a dedicated 70B parameter model for every individual creator. With PEFT / Multi-LoRA serving (e.g., vLLM or LoRAX), thousands of creator adapters can run concurrently on top of a single shared base model instance in GPU memory, routing incoming messages dynamically with zero GPU switching latency.

---

## 📌 Task 4: QLoRA Memory Monitoring & System RAM Analysis

The implementation is located in [`task_4_qlora_training_memory_monitor.py`](file:///d:/DA%26DS/fine_tuning/session_3_lora_peft_qlora/task_4_qlora_training_memory_monitor.py).

### Real-Time Memory Profiling:
- **Standard Fine-Tuning Peak Process RAM:** **2,346.24 MB**
- **QLoRA / PEFT Adaptation Peak Process RAM:** **1,450.48 MB**
- **RAM Savings:** **895.76 MB (38.2% RAM reduction)**

### Memory Comparison Chart:
The generated memory comparison chart is saved in [`assets/memory_usage_comparison.png`](file:///d:/DA%26DS/fine_tuning/assets/memory_usage_comparison.png).

![Memory Comparison Chart](file:///d:/DA%26DS/fine_tuning/assets/memory_usage_comparison.png)

### Key Insights:
In standard fine-tuning, the optimizer must allocate momentum and variance tensors for all 66.95 million parameters, causing memory usage to spike to 2.35 GB. Under PEFT / QLoRA, base weights are frozen and gradients are only computed for 739,586 parameters, capping peak RAM usage at 1.45 GB and preventing Out-Of-Memory (OOM) failures.

---

## 📌 Task 5: ChatGPT/Copilot LoRA Code Snippet & Environment Adaptations

The runnable script is located in [`task_5_chatgpt_lora_snippet.py`](file:///d:/DA%26DS/fine_tuning/session_3_lora_peft_qlora/task_5_chatgpt_lora_snippet.py).

### 1. Code Generated by ChatGPT / Copilot:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.to("cuda")
```

### 2. Changes Made to Get It Working in Local Environment:
1. **Target Modules Correction:** ChatGPT defaulted to `target_modules=["q_proj", "v_proj"]` (which applies to LLaMA/Mistral). For `distilbert-base-uncased`, attention projection layers are named `q_lin` and `v_lin`. Leaving `q_proj` caused an empty adapter initialization warning.
2. **Task Type Alignment:** ChatGPT specified `task_type="CAUSAL_LM"`. This was corrected to `TaskType.SEQ_CLS` so PEFT accurately preserves the sequence classification pooling head (`pre_classifier` and `classifier`) as trainable weights.
3. **Dynamic Device Handling:** Replaced hardcoded `.to("cuda")` with `torch.device("cuda" if torch.cuda.is_available() else "cpu")` to avoid runtime crashes on systems without dedicated GPU CUDA runtime.
