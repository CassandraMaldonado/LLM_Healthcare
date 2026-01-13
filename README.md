# Healthcare-Specialized LLM with Reinforcement Learning

**Healthcare-Specialized LLM with Parameter-Efficient Fine-Tuning, Preference Alignment and Memory-Augmented Reinforcement Learning**

University of Chicago - MS in Applied Data Science  
Cassandra Maldonado, Ke Wang, Jiayi Li, Zikai Wang

---

## Overview

Clinical decisions are often made under extreme time pressure. A physician may have **90 seconds** to act while a patient’s chart contains **hundreds of pages of notes and imaging reports**. While large language models (LLMs) offer speed and scale, they frequently hallucinate, drift across long contexts or produce unsafe clinical reasoning—making real-world deployment risky.

This project introduces **Clinical Q**, a healthcare-specialized LLM pipeline built on **LLaMA 3.1 (8B)**. We evaluate and combine **parameter-efficient fine-tuning (LoRA)**, **Direct Preference Optimization (DPO)**, **In-Context Learning (ICL)**, and a **memory-augmented method (Memento)** to improve factual accuracy, reasoning stability, and safety—without retraining full model weights.

When combined, **LoRA + Memento** produced the most reliable results overall:
- Higher factual accuracy  
- **65% fewer hallucinations**  
- Stronger generalization to unseen domains such as radiology reports  

---

## Why This Matters

- Diagnostic errors cause **$20B/year** in preventable harm.
- Manual clinical QA and documentation cost hospitals **$100K–$150K/year**.
- Hallucinations appear in **30–48%** of benchmark LLM outputs.
- Healthcare AI market projected to exceed **$300B by 2032**.

Clinical AI must be **fast, accurate, stable, and trustworthy**, not just fluent.

---

## Methods Evaluated

All methods are evaluated using the same **LLaMA 3.1 (8B)** base model, datasets, and metrics to ensure fair comparison.

### In-Context Learning (ICL)

The model receives structured clinical examples directly in the prompt, with **no parameter updates**.

**Role**
- Zero-training baseline  
- Measures reasoning ability from context alone  

**Limitations**
- Performance degrades with longer contexts  
- Higher hallucination rates  
- Limited generalization  

---

### Supervised Fine-Tuning with LoRA

LoRA fine-tunes a small set of low-rank adapter weights while keeping the base model frozen.

**Implementation**
- LoRA rank = 16  
- 8-bit quantization for memory efficiency  
- Curriculum-style training  

**Impact**
- Large gains in factual accuracy  
- Substantial hallucination reduction  
- Improved clinical tone and safety  

---

### Direct Preference Optimization (DPO)

DPO aligns the model with clinician-preferred reasoning using paired high- vs. low-quality responses.

**Impact**
- Improved safety and alignment  
- Reduced overconfidence  
- Moderate accuracy improvements  

---

### Memento: Memory-Augmented Learning

Memento augments the frozen model with a lightweight memory module that stores and reinforces prior reasoning traces using momentum-based updates.

**Impact**
- Improved long-context stability  
- Stronger multi-step reasoning  
- Reduced drift across cases  

---

### Clinical Q: LoRA + Memento

The final **Clinical Q** model combines LoRA fine-tuning with Memento memory augmentation.

**Why it works**
- LoRA injects domain knowledge efficiently  
- Memento stabilizes reasoning and consistency  
- Together they balance accuracy, safety, and generalization  

---

## Datasets

| Dataset | Purpose |
|------|------|
| **PubMedQA** | Evidence-based biomedical reasoning |
| **MedMCQA** | Broad factual medical knowledge |
| **MedQA (USMLE)** | Diagnostic reasoning and clinical decision-making |

These datasets jointly evaluate recall, reasoning depth, and safety.

---
