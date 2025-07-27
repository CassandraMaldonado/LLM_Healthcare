# Healthcare-Specialized LLM with Reinforcement Learning

## Project Overview
This repo explores how to fine-tune a healthcare-specialized language model using medical datasets like PubMedQA, RadQA, and MedMCQA. The goal was to improve performance on tasks like medical question answering and clinical reasoning, especially in situations where accuracy and safety really matter.

We also experimented with Reinforcement Learning with AI Feedback (RLAIF) to better align the model’s outputs with helpful, factual, and medically sound answers.

## Datasets
Here’s what I used and why:

_1. PubMedQA_
Biomedical yes/no/maybe questions from research abstracts

Great for evidence-based reasoning

_2. RadQA_
Radiology-focused QA with detailed rationales

Used to fine-tune on chain-of-thought reasoning

Really helpful for more step-by-step clinical logic

📖 MedMCQA
Massive MCQ dataset based on real medical exams (NEET-PG)

Covers a wide range of specialties

Good for factual knowledge and memorization-heavy tasks

Each dataset had a slightly different format and use case, so combining them helped balance reasoning, recall, and safety.
