# Fine-Tuning & Model Customization — Comprehensive Assignment Repository

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%2F%20QLoRA-green.svg)

This repository contains the complete session-wise implementations, analytical reports, empirical benchmarks, and interactive notebooks for the **Fine-Tuning & Model Customization** module of the Data Science & AI Program.

**Student:** Dhruv Munjpara  
**Repository:** [https://github.com/dhruv-munjpara/fine_tuning](https://github.com/dhruv-munjpara/fine_tuning)

---

## 📂 Repository Structure

```tree
d:/DA&DS/fine_tuning/
├── README.md                                             # Main documentation and guide
├── assets/
│   └── memory_usage_comparison.png                       # Empirical RAM benchmark plot (Standard vs QLoRA/PEFT)
├── session_1_introduction_and_why_finetune/
│   ├── session_1_answers.md                              # Deep-dive theory, domain scenarios, and decision matrix
│   ├── inspect_and_freeze_model.py                       # Task 4: Architecture inspection & base layer freezing
│   └── Session_1_Fine_Tuning_Overview.ipynb              # Interactive notebook with outputs
├── session_2_prompt_vs_rag_vs_finetuning/
│   ├── session_2_answers.md                              # Trade-off analysis, app case studies, BookMyShow decision
│   ├── ipl_stats.txt                                     # Curated IPL cricket statistics knowledge corpus
│   ├── simulate_rag_workflow.py                          # Task 2: Grounded RAG workflow simulator
│   ├── prompt_rag_finetune_comparison.csv                # Structured comparative matrix spreadsheet
│   └── Session_2_Prompt_vs_RAG_vs_FineTuning.ipynb       # Interactive notebook with outputs
└── session_3_lora_peft_qlora/
    ├── session_3_answers.md                              # PEFT advantages, multi-tenant serving, QLoRA analysis
    ├── task_1_lora_trainable_params.py                   # Task 1: LoRA parameter reduction calculation
    ├── task_2_finetune_lora_custom_dataset.py            # Task 2: Sentiment classification training on 10 tweets
    ├── task_4_qlora_training_memory_monitor.py           # Task 4: System RAM monitoring & profiling
    ├── task_5_chatgpt_lora_snippet.py                    # Task 5: Adapted ChatGPT snippet & environment fixes
    ├── Session_3_LoRA_PEFT_QLoRA.ipynb                   # Interactive notebook with outputs
    └── saved_lora_adapter/                               # Saved LoRA adapter weights (~3.5 MB footprint)
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## 📑 Session Breakdown & Key Findings

### 🔹 Session 1: Fine-Tuning Introduction & Why Fine-Tune
* **Pre-Training vs Fine-Tuning:** Pre-training builds general statistical next-token capability over web-scale text, whereas fine-tuning aligns the model to task instructions, specialized vocabularies, and deterministic output schemas.
* **Domain Failure Scenarios:** Detailed case studies for **Healthcare** (EHR/ICD-10 clinical coding), **Legal** (statutory precision in contracts), and **Quantitative Finance** (counter-intuitive sentiment in SEC filings).
* **Architecture Inspection & Freezing ([`inspect_and_freeze_model.py`](session_1_introduction_and_why_finetune/inspect_and_freeze_model.py)):**
  - Base Model: `distilbert-base-uncased` (66,955,010 total parameters).
  - Trainable parameters before freezing: **66,955,010 (100.00%)**.
  - Trainable parameters after freezing base transformer: **592,130 (0.88%)**.
  - Frozen base parameters: **66,362,880 (99.12%)**.
* **Decision Framework:** Clear criteria for when **not** to fine-tune (dynamic data, need for exact citations, limited training examples).

---

### 🔹 Session 2: Prompt Engineering vs RAG vs Fine-Tuning
* **5-Factor Evaluation:** Comparative evaluation across Data Freshness, Hallucination Control, Setup Cost, Latency, and Customization Depth.
* **IPL Cricket Stats RAG Simulation ([`simulate_rag_workflow.py`](session_2_prompt_vs_rag_vs_finetuning/simulate_rag_workflow.py)):**
  - Ingests structured facts from [`ipl_stats.txt`](session_2_prompt_vs_rag_vs_finetuning/ipl_stats.txt).
  - Implements token scoring, entity boosting (e.g. *Virat Kohli, CSK, SRH*), context augmentation, and factual grounded synthesis without hallucinations.
* **Real-World App Scenarios:**
  - **Zomato:** Prompt Engineering for dynamic dining suggestions based on session cart and mood.
  - **Flipkart:** RAG for live order tracking, return window verification, and product policy search.
  - **Paytm:** Fine-Tuning compact models for sub-10ms UPI fraud and anomaly detection.
* **Structured Comparison Table:** Saved in [`prompt_rag_finetune_comparison.csv`](session_2_prompt_vs_rag_vs_finetuning/prompt_rag_finetune_comparison.csv).
* **BookMyShow Chatbot Evaluation:** Recommended **Hybrid RAG + Tool Calling** to support real-time theater showtimes and dynamic seat availability.

---

### 🔹 Session 3: LoRA, PEFT & QLoRA Deep Dive
* **LoRA Trainable Parameters ([`task_1_lora_trainable_params.py`](session_3_lora_peft_qlora/task_1_lora_trainable_params.py)):**
  - Base parameters: 66,955,010.
  - Trainable parameters with LoRA ($r=8, \alpha=16$): **739,586 (1.0925%)**.
  - Parameter reduction: **98.90%**.
* **Fine-Tuning on Custom Sentiment Dataset ([`task_2_finetune_lora_custom_dataset.py`](session_3_lora_peft_qlora/task_2_finetune_lora_custom_dataset.py)):**
  - Fine-tuned on 10 custom labeled product review tweets for 4 epochs (loss dropped from 0.8793 $\rightarrow$ 0.3158).
  - Saved adapter weights to [`saved_lora_adapter/`](session_3_lora_peft_qlora/saved_lora_adapter).
  - Disk footprint: `adapter_model.safetensors` is **2.89 MB** (total folder ~3.51 MB) vs **~268 MB** for a full model checkpoint (**~98.7% disk savings**).
* **Memory Footprint Benchmark ([`task_4_qlora_training_memory_monitor.py`](session_3_lora_peft_qlora/task_4_qlora_training_memory_monitor.py)):**
  - Standard Fine-Tuning Peak Process RAM: **2,346.24 MB**
  - QLoRA / PEFT Adaptation Peak Process RAM: **1,450.48 MB**
  - Memory reduction: **895.76 MB (38.2% RAM savings)**.
  - Visual chart saved at [`assets/memory_usage_comparison.png`](assets/memory_usage_comparison.png).
* **ChatGPT Code Snippet Adaptation ([`task_5_chatgpt_lora_snippet.py`](session_3_lora_peft_qlora/task_5_chatgpt_lora_snippet.py)):**
  - Corrected target modules from `q_proj, v_proj` to `q_lin, v_lin` for DistilBERT.
  - Changed task type to `TaskType.SEQ_CLS` to keep classification head trainable.
  - Added dynamic device selection (`cuda` vs `cpu`).

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch transformers peft accelerate pandas scikit-learn matplotlib psutil bitsandbytes
```

### 2. Run Session 1
```bash
python session_1_introduction_and_why_finetune/inspect_and_freeze_model.py
```

### 3. Run Session 2
```bash
python session_2_prompt_vs_rag_vs_finetuning/simulate_rag_workflow.py
```

### 4. Run Session 3
```bash
python session_3_lora_peft_qlora/task_1_lora_trainable_params.py
python session_3_lora_peft_qlora/task_2_finetune_lora_custom_dataset.py
python session_3_lora_peft_qlora/task_4_qlora_training_memory_monitor.py
python session_3_lora_peft_qlora/task_5_chatgpt_lora_snippet.py
```
