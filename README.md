# Healthcare-Specialized LLM with Reinforcement Learning

## Overview
This project focuses on fine-tuning a biomedical LLM for clinical question answering. The main goal is to improve factual accuracy, diagnostic reasoning, and clinical safety by training on high-quality medical datasets.

The pipeline uses multi-stage fine-tuning with LoRA adapters and 8-bit quantization, making it efficient to run while still effective for healthcare-related tasks.

## Motivation

General-purpose LLMs often struggle in clinical settings because they:

- Miss medical nuance.

- Sometimes hallucinate answers.

- Lack domain-specific context.

By using domain-specific datasets and reinforcement alignment techniques, this project aims to train models that provide safer and more reliable medical responses.

## Datasets

Each dataset contributes something different to the training:

PubMedQA – Biomedical yes/no/maybe questions from PubMed abstracts. Good for evidence-based reasoning.

RadQA – Radiology QA with detailed rationales. Helps the model learn step-by-step (chain-of-thought) clinical logic.

MedMCQA – A large multiple-choice dataset based on medical exams. Useful for memorization and broad factual knowledge.

Combining these datasets helps balance reasoning, recall, and safety in the final model.

## Summary
Most general LLMs aren't reliable enough for clinical use they miss nuance, sometimes hallucinate and lack medical context. So for this project, I:

- Fine-tuned a base model (LLaMA) using three healthcare datasets.

- Trained a reward model to score helpfulness and safety.

- Applied reinforcement learning to align responses with expert like answers.

- Evaluated the improvements across different medical QA formats.
