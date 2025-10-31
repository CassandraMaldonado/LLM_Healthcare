# Healthcare-Specialized LLM with Reinforcement Learning

## Overview

This project explores three small experiments to compare different ways of adapting LLaMA 3.1 for medical question answering. The goal is to test and compare fine-tuning, ICL and DPO using a small sample of medical data to see which approach gives the best balance between accuracy, reasoning quality and safety.

Instead of training one big model, we’re running smaller, controlled tests using about 100 examples per method. The results will help us decide which strategy is most effective before scaling up for larger experiments.

## Motivation

General LLMs often struggle with biomedical content they sometimes hallucinate, overlook key clinical details or produce unsafe advice. Because training big models requires significant compute power, we’re starting small to figure out which method of tuning LLaMA 3.1 is most effective for healthcare tasks.

## Experiments

Each experiment uses the same LLaMA 3.1 base model and a consistent set of 100 medical question–answer pairs for fair comparison.

***1. Fine-Tuning***

We fine-tune LLaMA 3.1 using supervised fine-tuning (SFT) with LoRA adapters. This method trains only a small portion of the model’s parameters, making it lightweight and efficient. The goal is to teach the model directly from labeled medical QA pairs.

***2. In-Context Learning (ICL)***

In this setup, the model isn’t trained, it just sees examples directly in the prompt. This tests how well LLaMA 3.1 can learn patterns and reasoning styles from context alone, without updating its weights. 

**_3. Direct Preference Optimization (DPO)_**

DPO helps the model learn to prefer higher-quality answers. Using pairs of good vs. less-good responses, the model adjusts its behavior to align with human-preferred clinical reasoning and tone.

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


## Results and Evaluation

**_1. Baseline Model_**

The baseline experiment used the original **LLaMA 3.1 8B** model without any healthcare-specific adaptation. The evaluation was performed using the same datasets (PubMedQA, RadQA, MedMCQA) to establish reference metrics.

| Metric | PubMedQA | MedMCQA | RadQA | Notes |
|:--|:--:|:--:|:--:|:--|
| Accuracy | 61.4% | 58.7% | 55.2% | The model performs decently on factual tasks but struggles with reasoning depth. |
| Precision / Recall | 0.59 / 0.61 | 0.57 / 0.59 | 0.53 / 0.54 | It indicates a surface level understanding. |
| Hallucination rate | 14% | – | – | Frequent unsupported claims in complex prompts. |


**_2. Fine-Tuned Model_**

The fine-tuned version used a three-stage curriculum (PubMedQA -> MedMCQA → MedQA)** with **LoRA adapters** at rank = 16 and **8-bit quantization**.  
Training logs and metrics were tracked using Weights & Biases.

| Metric | PubMedQA | MedMCQA | RadQA | Improvement vs Baseline |
|:--|:--:|:--:|:--:|:--|
| Accuracy | 78.9 % | 74.5% | 69.3% | +17 pp gain across datasets |
| Precision / Recall | 0.78 / 0.77 | 0.75 / 0.74 | 0.70 / 0.69 | Balanced improvements in reasoning and recall |
| Hallucination Rate | decreased to 5 % | – | – | Strong reduction in unsupported statements |
| Safety Score (Reward Model) | ↑ to 0.89 | – | – | Better alignment with clinical tone and caution |

These results confirm that lightweight parameter-efficient fine-tuning can significantly improve factual accuracy, reasoning coherence, and safety without retraining the full model.


**_3. RAGAS Evaluation (Retrieval-Augmented Generation)_**

A complementary RAGAS evaluation pipeline was added to quantify factuality and contextual precision of model outputs.  
This notebook measures **Faithfulness**, **Answer Relevance**, **Context Precision**, and **Context Recall** for both the base and fine-tuned models.

| Metric | Baseline(%) | Fine-Tuned(%) | Change |
|:--|:--:|:--:|:--:|
| Faithfulness | 72% | 91% | +19 pp |
| Answer Relevance | 68% | 88% | +20 pp |
| Context Precision | 64% | 85% | +21 pp |
| Context Recall | 70% | 89% | +19 pp |

These RAGAS scores indicate that fine-tuning improved not only accuracy but also contextual grounding — the model now generates answers that are more faithful to the retrieved evidence.

---

### 4. Key Takeaways

- Fine-tuning yields **stronger domain reasoning** and **fewer hallucinations** compared to prompting alone.  
- LoRA + quantization allows **efficient training** on limited hardware.  
- RAGAS evaluation highlights improved **context-evidence alignment**, validating both retrieval and generation quality.  
- The pipeline provides a reproducible foundation for scaling to larger datasets or reinforcement-learning stages.

---

### 📁 Notebooks Reference

| Notebook | Purpose |
|:--|:--|
| `baseline_model_results.ipynb` | Establishes baseline metrics for the un-tuned LLaMA 3.1 model. |
| `finetuned_model_results.ipynb` | Contains evaluation logs and metric comparisons for the LoRA-tuned model. |
| `Ragas_eval.ipynb` | Implements RAGAS metric evaluation for factuality and context grounding. |

---

### 🚀 Next Steps

- Extend evaluation to **clinical-dialogue datasets** (e.g., MedDialog).  
- Apply **Direct Preference Optimization (DPO)** for alignment refinement.  
- Integrate a **retrieval-augmented inference layer** for evidence citation during response generation.

---



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

- Validating data formatting (prompt/response structure, tokenization) quickly.

- Iterating prompt templates and instruction tuning before scaling to 13B.

- Running small-scale inference and sanity checks on hardware that can’t handle the full 13B workflow.

## Summary
Most general LLMs aren't reliable enough for clinical use they miss nuance, sometimes hallucinate and lack medical context. So for this project, I:

- Fine-tuned a base model (LLaMA) using three healthcare datasets.

- Trained a reward model to score helpfulness and safety.

- Applied reinforcement learning to align responses with expert like answers.

- Evaluated the improvements across different medical QA formats.
