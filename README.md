# Healthcare-Specialized LLM with Reinforcement Learning

## Overview
This project fine-tunes a biomedical large language model (LLM) for clinical question answering. The goal is to improve factual correctness, diagnostic reasoning, and safety in responses, making the model more suitable for use in healthcare settings.

The training pipeline uses multi-stage curriculum fine-tuning with LoRA adapters and 8-bit quantization, which allows large models to be trained efficiently while maintaining strong performance.

## Motivation

Most general LLMs struggle in biomedical contexts: they hallucinate, miss subtle clinical cues, and don’t generalize well to exam-style or radiology reasoning tasks. To address this, I fine-tuned an open-source LLM using carefully selected medical datasets, aligning the model step by step with expert-like reasoning and safe clinical behavior.

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

Consistency on PubMedQA (yes/no/maybe biomedical evidence questions).

In addition, I monitored:

Hallucination rate – fewer unsupported claims after fine-tuning.

Safety alignment – responses scored for clinical appropriateness using a reward model.

Comparisons to baseline – fine-tuned model vs. base model performance.

## Summary
Most general LLMs aren't reliable enough for clinical use they miss nuance, sometimes hallucinate and lack medical context. So for this project, I:

- Fine-tuned a base model (LLaMA) using three healthcare datasets.

- Trained a reward model to score helpfulness and safety.

- Applied reinforcement learning to align responses with expert like answers.

- Evaluated the improvements across different medical QA formats.
