# Healthcare-Specialized LLM with Reinforcement Learning

## Overview

This project explores three small experiments to compare different ways of adapting LLaMA 3.1 for medical question answering. The goal is to test and compare fine-tuning, ICL and DPO using a small sample of medical data to see which approach gives the best balance between accuracy, reasoning quality and safety.

Instead of training one big model, we’re running smaller, controlled tests using about 100 examples per method. The results will help us decide which strategy is most effective before scaling up for larger experiments.

## Motivation

General LLMs often struggle with biomedical content they sometimes hallucinate, overlook key clinical details or produce unsafe advice. Because training big models requires significant compute power, we’re starting small to figure out which method of tuning LLaMA 3.1 is most effective for healthcare tasks.

## Experiments

Each experiment uses the same small base model (like LLaMA or GPT-OSS) and a consistent 50-example dataset for fair comparison.

1. Fine-Tuning

We train the model using supervised fine-tuning (SFT) with LoRA adapters. Only a small set of parameters is updated, which saves memory and training time. This helps the model learn directly from labeled question–answer pairs.

2. In-Context Learning (ICL)

In this setup, we don’t train the model at all — we just show it examples directly in the prompt and see how well it generalizes. This test helps us understand how much the model can learn “on the fly” from context alone.

3. Direct Preference Optimization (DPO)

This experiment teaches the model to prefer better-quality answers. We use pairs of responses — one preferred, one not — so the model learns what kind of reasoning and tone are most clinically appropriate.

## Datasets

Each dataset contributes something different to the training:

1. PubMedQA: Biomedical yes/no/maybe questions from PubMed abstracts. Good for evidence-based reasoning.

2. RadQA: Radiology QA with detailed rationales. Helps the model learn step-by-step (chain-of-thought) clinical logic.

3. MedMCQA: A large multiple-choice dataset based on medical exams. Useful for memorization and broad factual knowledge.

Combining these datasets helps balance reasoning, recall, and safety in the final model.

## Fine-Tuning Process

The fine-tuning pipeline is implemented in 'finetune_multistep.py'. The process follows a curriculum-style setup:

**1. Instruction Formatting:** Converted QA datasets into an instruction–response format so the model learns to follow prompts directly.

**2. Stage-Wise Training:**

- _Stage 1:_ Fine-tune on PubMedQA for evidence reasoning.

- _Stage 2:_ Fine-tune on MedMCQA for factual recall.

- _Stage 3:_ Fine-tune on MedQA for comprehensive clinical reasoning.

**3. LoRA Adapters:** Used for parameter-efficient fine-tuning, training only a fraction of model weights.

**4. Quantization:** Trained with 8-bit precision to reduce GPU memory usage and make large models trainable on more modest hardware.

**5.Experiment Tracking:** Integrated with W&B for logging losses, evaluation metrics and checkpoints.

## Evaluation

To measure improvements, I evaluated the fine-tuned model across different QA formats:

- Accuracy on MedMCQA (factual exam-style questions).

- Reasoning Quality on RadQA (step-by-step CoT explanations).

- Consistency on PubMedQA (yes/no/maybe biomedical evidence questions).

In addition, I monitored:

- **Hallucination rate:** fewer unsupported claims after fine-tuning.

- **Safety alignment:** responses scored for clinical appropriateness using a reward model.

- **Comparisons to baseline:** fine-tuned model vs. base model performance.

## Notebooks (in Gpt-oss)

These notebooks provide runnable workflows to explore and test the models interactively. They complement the production scripts by giving you a more hands-on way to understand and validate each training stage.

- **LLama_13b.ipynb**
  
A comprehensive fine-tuning walkthrough for the 13B-parameter LLaMA model.
It covers:

1. Loading a quantized version of the 13B model (8-bit or 4-bit modes).

2. Attaching LoRA adapters to the model, freezing base weights.

3. Sequential fine-tuning stages (PubMedQA, RadQA, MedMCQA) with intermediate checkpointing.

4. Running evaluations after each stage (accuracy, reasoning, safety metrics).

5. Sampling and comparing model outputs across prompts to monitor hallucination or alignment issues.

- **Testing_llama7b.ipynb**

A quicker, lighter alternative using the 7B-parameter model.
Use cases include:

Validating data formatting (prompt/response structure, tokenization) quickly.

Iterating prompt templates and instruction tuning before scaling to 13B.

Running small-scale inference and sanity checks on hardware that can’t handle the full 13B workflow.

## Summary
Most general LLMs aren't reliable enough for clinical use they miss nuance, sometimes hallucinate and lack medical context. So for this project, I:

- Fine-tuned a base model (LLaMA) using three healthcare datasets.

- Trained a reward model to score helpfulness and safety.

- Applied reinforcement learning to align responses with expert like answers.

- Evaluated the improvements across different medical QA formats.
