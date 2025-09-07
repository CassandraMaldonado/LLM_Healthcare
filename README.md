# Healthcare-Specialized LLM with Reinforcement Learning

## Overview
This project focuses on fine-tuning a biomedical LLM for clinical question answering. The main goal is to improve factual accuracy, diagnostic reasoning, and clinical safety by training on high-quality medical datasets.

The pipeline uses multi-stage fine-tuning with LoRA adapters and 8-bit quantization, making it efficient to run while still effective for healthcare-related tasks.

## Datasets
Here’s what I used and why:

_1. PubMedQA_: 
Biomedical yes/no/maybe questions from research abstracts, which makes it great for evidence reasoning.

_2. RadQA_: 
Radiology QA with detailed rationales, used to fine-tune on CoT reasoning. Really helpful for more step-by-step clinical logic.

_3. MedMCQA_: 
Massive MCQ dataset based on real medical exams, it covers a wide range of specialties. Good for factual knowledge and memorization tasks.

Each dataset had a slightly different format and use case, so combining them helped balance reasoning, recall and safety.

## Summary
Most general LLMs aren't reliable enough for clinical use they miss nuance, sometimes hallucinate and lack medical context. So for this project, I:

- Fine-tuned a base model (LLaMA) using three healthcare datasets.

- Trained a reward model to score helpfulness and safety.

- Applied reinforcement learning to align responses with expert like answers.

- Evaluated the improvements across different medical QA formats.
