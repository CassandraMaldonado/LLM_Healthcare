# Evaluation script for fine-tuned medical QA models.

!pip install rouge-score nltk scikit-learn bert-score spacy textstat pandas numpy
python -m spacy download en_core_web_sm

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re

try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    ADVANCED_METRICS = True
except ImportError:
    print("Warning: ROUGE and NLTK not installed.")
    ADVANCED_METRICS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    accuracy: float = 0.0
    exact_match: float = 0.0
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    avg_response_length: float = 0.0
    total_samples: int = 0
    inference_time_per_sample: float = 0.0
    detailed_results: List[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "exact_match": self.exact_match,
            "bleu_score": self.bleu_score,
            "rouge_l": self.rouge_l,
            "rouge_1": self.rouge_1,
            "rouge_2": self.rouge_2,
            "avg_response_length": self.avg_response_length,
            "total_samples": self.total_samples,
            "inference_time_per_sample": self.inference_time_per_sample,
        }


class MedicalQAEvaluator:
    
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "mistralai/gpt-oss-20b",
        use_8bit: bool = True,
        max_length: int = 2048,
        max_new_tokens: int = 512,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.use_8bit = use_8bit
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device
        
        if ADVANCED_METRICS:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Model and tokenizer.
        self.model, self.tokenizer = self._load_model()

# Fine-tuned model and tokenizer.      
    def _load_model(self) -> Tuple[torch.nn.Module, Any]:
        logger.info(f"Loading model from {self.model_path}")
        
        quantization_config = None
        if self.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        
        # Tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Checking if is a PEFT model or full model.
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            # PEFT model.
            logger.info("Loading as PEFT model.")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Full model.
            logger.info("Loading as full model.")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        model.eval()
        return model, tokenizer
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        # Tokenizing the input.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - self.max_new_tokens,
            padding=True,
        )
        

        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation parameters.
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **generation_kwargs
        }
        

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        
        # Decoding response.
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

# Evaluation dataset.    
    def load_evaluation_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} evaluation samples from {data_path}")
        return data

# Exact match score.    
    def calculate_exact_match(self, predicted: str, reference: str) -> float:
        pred_normalized = predicted.lower().strip()
        ref_normalized = reference.lower().strip()
        return float(pred_normalized == ref_normalized)

# BLEU score.   
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        if not ADVANCED_METRICS:
            return 0.0
        
        try:
            # Tokenize.
            pred_tokens = predicted.lower().split()
            ref_tokens = [reference.lower().split()]
            
            # BLEU with smoothing.
            smoothie = SmoothingFunction().method4
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
            return score
        except:
            return 0.0

# ROUGE scores. 
    def calculate_rouge_scores(self, predicted: str, reference: str) -> Dict[str, float]:
        if not ADVANCED_METRICS:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }
        except:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

# Extracting the answer choice from both predicted and reference.    
    def calculate_accuracy_for_mcqa(self, predicted: str, reference: str, choices: List[str] = None) -> float:
        pred_choice = self.extract_choice(predicted)
        ref_choice = self.extract_choice(reference)
        
        if pred_choice and ref_choice:
            return float(pred_choice.upper() == ref_choice.upper())
        
        return self.calculate_exact_match(predicted, reference)
    
    def extract_choice(self, text: str) -> Optional[str]:
        """Extract answer choice (A, B, C, D) from text."""
        patterns = [
            r'\b([ABCDE])\)',
            r'\(([ABCDE])\)',
            r'\b([ABCDE])\.',
            r'(?:answer|choice):\s*([ABCDE])',
            r'(?:the answer is|answer is)\s*([ABCDE])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.upper())
            if match:
                return match.group(1)
        
        text_clean = text.strip().upper()
        if text_clean and text_clean[0] in 'ABCDE':
            return text_clean[0]
        
        return None
    
    def evaluate_sample(self, sample: Dict, sample_idx: int) -> Dict:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")
        
        # Formatting the prompt.
        prompt = self.format_prompt(instruction, input_text)
        
        # Response.
        start_time = time.time()
        predicted = self.generate_response(prompt)
        inference_time = time.time() - start_time
        
        # Metrics.
        exact_match = self.calculate_exact_match(predicted, reference)
        bleu_score = self.calculate_bleu_score(predicted, reference)
        rouge_scores = self.calculate_rouge_scores(predicted, reference)
        
        # For multiple choice questions we use specialized accuracy.
        choices = sample.get("choices", [])
        if choices or any(choice in instruction.lower() for choice in ['a)', 'b)', 'c)', 'd)']):
            accuracy = self.calculate_accuracy_for_mcqa(predicted, reference, choices)
        else:
            accuracy = exact_match
        
        result = {
            "sample_idx": sample_idx,
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "predicted": predicted,
            "exact_match": exact_match,
            "accuracy": accuracy,
            "bleu_score": bleu_score,
            "rouge_1": rouge_scores["rouge1"],
            "rouge_2": rouge_scores["rouge2"],
            "rouge_l": rouge_scores["rougeL"],
            "response_length": len(predicted.split()),
            "inference_time": inference_time,
        }
        
        return result
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_dir: str = "./evaluation_results",
        max_samples: Optional[int] = None,
        save_detailed: bool = True,
    ) -> EvaluationResults:
        logger.info(f"Starting evaluation on {data_path}")
        
        # Data.
        data = self.load_evaluation_data(data_path)
        if max_samples:
            data = data[:max_samples]
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluating each sample.
        detailed_results = []
        for i, sample in enumerate(tqdm(data, desc="Evaluating samples")):
            try:
                result = self.evaluate_sample(sample, i)
                detailed_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
        
        # Aggregate metrics.
        if not detailed_results:
            logger.error("No samples were evaluated.")
            return EvaluationResults()
        
        # Separating MCQ and non-MCQ samples.
        mcq_results = [r for r in detailed_results if r["is_mcq"]]
        non_mcq_results = [r for r in detailed_results if not r["is_mcq"]]
        
        # Core metrics.
        accuracy = np.mean([r["accuracy"] for r in detailed_results])
        exact_match = np.mean([r["exact_match"] for r in detailed_results])
        
        # F1, Precision and Recall for binary classification.
        y_true = [1 if r["accuracy"] > 0.5 else 0 for r in detailed_results]
        y_pred = [1 if r["accuracy"] > 0.5 else 0 for r in detailed_results]
        
        if ADVANCED_METRICS and len(set(y_true)) > 1:
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        else:
            f1 = accuracy
            precision = accuracy
            recall = accuracy
        
        # BLEU scores.
        bleu_1 = np.mean([r["bleu_1"] for r in detailed_results])
        bleu_2 = np.mean([r["bleu_2"] for r in detailed_results])
        bleu_3 = np.mean([r["bleu_3"] for r in detailed_results])
        bleu_4 = np.mean([r["bleu_4"] for r in detailed_results])
        
        # ROUGE scores.
        rouge_1 = np.mean([r["rouge_1"] for r in detailed_results])
        rouge_2 = np.mean([r["rouge_2"] for r in detailed_results])
        rouge_l = np.mean([r["rouge_l"] for r in detailed_results])
        
        # Semantic similarity.
        semantic_similarity = np.mean([r["semantic_similarity"] for r in detailed_results])
        
        # BERTScore.
        bertscore_f1 = np.mean([r["bertscore_f1"] for r in detailed_results])
        bertscore_precision = np.mean([r["bertscore_precision"] for r in detailed_results])
        bertscore_recall = np.mean([r["bertscore_recall"] for r in detailed_results])
        
        # Medical metrics.
        clinical_accuracy = np.mean([r["clinical_accuracy"] for r in detailed_results])
        factual_correctness = np.mean([r["factual_correctness"] for r in detailed_results])
        safety_score = np.mean([r["safety_score"] for r in detailed_results])
        
        # Readability.
        flesch_reading_ease = np.mean([r["flesch_reading_ease"] for r in detailed_results])
        flesch_kincaid_grade = np.mean([r["flesch_kincaid_grade"] for r in detailed_results])
        gunning_fog = np.mean([r["gunning_fog"] for r in detailed_results])
        
        # MCQ specific.
        mcq_accuracy = np.mean([r["mcq_accuracy"] for r in mcq_results]) if mcq_results else 0.0
        mcq_confidence = np.mean([r["mcq_confidence"] for r in mcq_results]) if mcq_results else 0.0
        
        # Response characteristics.
        avg_word_count = np.mean([r["word_count"] for r in detailed_results])
        avg_sentence_count = np.mean([r["sentence_count"] for r in detailed_results])
        avg_response_length = np.mean([r["response_length"] for r in detailed_results])
        avg_inference_time = np.mean([r["inference_time"] for r in detailed_results])
        
        # Answer distribution analysis.
        if mcq_results:
            predicted_choices = []
            for r in mcq_results:
                choice = self.extract_choice(r["predicted"])
                if choice:
                    predicted_choices.append(choice)
            
            if predicted_choices:
                from collections import Counter
                choice_counts = Counter(predicted_choices)
                total_mcq = len(predicted_choices)
                answer_distribution = {choice: count/total_mcq for choice, count in choice_counts.items()}
            else:
                answer_distribution = {}
        else:
            answer_distribution = {}
        
        # Unanswered rate.
        unanswered_count = sum(1 for r in detailed_results if r["is_unanswered"])
        unanswered_rate = unanswered_count / len(detailed_results)
        
        results = EvaluationResults(
            # Core metrics.
            accuracy=accuracy,
            exact_match=exact_match,
            f1_score=f1,
            precision=precision,
            recall=recall,
            
            # BLEU scores.
            bleu_1=bleu_1,
            bleu_2=bleu_2,
            bleu_3=bleu_3,
            bleu_4=bleu_4,
            
            # ROUGE scores.
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            
            # Semantic similarity.
            semantic_similarity=semantic_similarity,
            bertscore_f1=bertscore_f1,
            bertscore_precision=bertscore_precision,
            bertscore_recall=bertscore_recall,
            
            # Medical metrics.
            clinical_accuracy=clinical_accuracy,
            factual_correctness=factual_correctness,
            safety_score=safety_score,
            
            # Readability.
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog=gunning_fog,
            
            # MCQ specific.
            mcq_accuracy=mcq_accuracy,
            mcq_confidence=mcq_confidence,
            
            # Response characteristics.
            avg_response_length=avg_response_length,
            avg_word_count=avg_word_count,
            avg_sentence_count=avg_sentence_count,
            total_samples=len(detailed_results),
            inference_time_per_sample=avg_inference_time,
            answer_distribution=answer_distribution,
            unanswered_rate=unanswered_rate,
            detailed_results=detailed_results if save_detailed else None,
        )results])
        rouge_l = np.mean([r["rouge_l"] for r in detailed_results])
        avg_response_length = np.mean([r["response_length"] for r in detailed_results])
        avg_inference_time = np.mean([r["inference_time"] for r in detailed_results])
        
        results = EvaluationResults(
            accuracy=accuracy,
            exact_match=exact_match,
            bleu_score=bleu_score,
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            avg_response_length=avg_response_length,
            total_samples=len(detailed_results),
            inference_time_per_sample=avg_inference_time,
            detailed_results=detailed_results if save_detailed else None,
        )
        
        # Results.
        dataset_name = Path(data_path).stem
        
        # Summary.
        summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        if save_detailed:
            detailed_path = os.path.join(output_dir, f"{dataset_name}_detailed.json")
            with open(detailed_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            df = pd.DataFrame(detailed_results)
            csv_path = os.path.join(output_dir, f"{dataset_name}_detailed.csv")
            df.to_csv(csv_path, index=False)
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return results


def print_results(results: EvaluationResults, dataset_name: str):
    print(f"Evaluation Results {dataset_name.upper()}")
    print(f"{'-'*60}")
    print(f"Total Samples: {results.total_samples}")
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"Exact Match: {results.exact_match:.4f}")
    if ADVANCED_METRICS:
        print(f"BLEU Score: {results.bleu_score:.4f}")
        print(f"ROUGE-1: {results.rouge_1:.4f}")
        print(f"ROUGE-2: {results.rouge_2:.4f}")
        print(f"ROUGE-L: {results.rouge_l:.4f}")
    print(f"Avg Response Length: {results.avg_response_length:.1f} words")
    print(f"Avg Inference Time: {results.inference_time_per_sample:.3f} seconds")
    print(f"{'-'*60}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive medical QA model evaluation")
    
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", default="mistralai/gpt-oss-20b", help="Base model name")
    parser.add_argument("--eval_data", required=True, help="Path to evaluation dataset (JSONL)")
    parser.add_argument("--output_dir", default="./evaluation_results", help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--use_8bit", action="store_true", default=True, help="Use 8-bit quantization")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--no_detailed", action="store_true", help="Don't save detailed results")
    parser.add_argument("--batch_eval", nargs='+', help="Evaluate multiple datasets")
    parser.add_argument("--compare_models", nargs='+', help="Compare multiple model checkpoints")
    
    args = parser.parse_args()
    
    if args.batch_eval:
        # Batch mode.
        logger.info(f"Batch evaluation mode: evaluating {len(args.batch_eval)} datasets")
        all_results = {}
        
        evaluator = MedicalQAEvaluator(
            model_path=args.model_path,
            base_model_name=args.base_model,
            use_8bit=args.use_8bit,
            max_new_tokens=args.max_new_tokens,
        )
        
        for dataset_path in args.batch_eval:
            dataset_name = Path(dataset_path).stem
            logger.info(f"Evaluating on {dataset_name}")
            
            results = evaluator.evaluate_dataset(
                data_path=dataset_path,
                output_dir=os.path.join(args.output_dir, dataset_name),
                max_samples=args.max_samples,
                save_detailed=not args.no_detailed,
            )
            
            all_results[dataset_name] = results
            print_results(results, dataset_name)
        
        # Summary.
        comparison_summary = {}
        for name, results in all_results.items():
            comparison_summary[name] = results.to_dict()
        
        with open(os.path.join(args.output_dir, "batch_comparison.json"), 'w') as f:
            json.dump(comparison_summary, f, indent=2)
            
    elif args.compare_models:
        # Model comparison mode.
        logger.info(f"Model comparison mode: comparing {len(args.compare_models)} models.")
        all_results = {}
        
        for model_path in args.compare_models:
            model_name = Path(model_path).name
            logger.info(f"Evaluating model: {model_name}")
            
            evaluator = MedicalQAEvaluator(
                model_path=model_path,
                base_model_name=args.base_model,
                use_8bit=args.use_8bit,
                max_new_tokens=args.max_new_tokens,
            )
            
            results = evaluator.evaluate_dataset(
                data_path=args.eval_data,
                output_dir=os.path.join(args.output_dir, model_name),
                max_samples=args.max_samples,
                save_detailed=not args.no_detailed,
            )
            
            all_results[model_name] = results
            print_results(results, model_name)
        
        # Model comparison.
        comparison_summary = {}
        for name, results in all_results.items():
            comparison_summary[name] = results.to_dict()
        
        with open(os.path.join(args.output_dir, "model_comparison.json"), 'w') as f:
            json.dump(comparison_summary, f, indent=2)
    
    else:
        # Single evaluation mode.
        evaluator = MedicalQAEvaluator(
            model_path=args.model_path,
            base_model_name=args.base_model,
            use_8bit=args.use_8bit,
            max_new_tokens=args.max_new_tokens,
        )
        
        # Parameters.
        generation_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }
        
        # Overriding the generation method to use custom parameters.
        original_generate = evaluator.generate_response
        def generate_with_params(prompt, **kwargs):
            return original_generate(prompt, **{**generation_kwargs, **kwargs})
        evaluator.generate_response = generate_with_params
        
        results = evaluator.evaluate_dataset(
            data_path=args.eval_data,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            save_detailed=not args.no_detailed,
        )
        
        dataset_name = Path(args.eval_data).stem
        print_results(results, dataset_name)

        
        summary_file = os.path.join(args.output_dir, "evaluation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Medical QA LLM Evaluation Report.")
            f.write(f"{'-'*50}\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Dataset: {args.eval_data}\n")
            f.write(f"Total Samples: {results.total_samples}\n\n")
            
            f.write(f"Core metrics:\n")
            f.write(f"- Overall Accuracy: {results.accuracy:.4f}\n")
            f.write(f"- Exact Match: {results.exact_match:.4f}\n")
            f.write(f"- F1 Score: {results.f1_score:.4f}\n")
            f.write(f"- Precision: {results.precision:.4f}\n")
            f.write(f"- Recall: {results.recall:.4f}\n\n")
            
            f.write(f"Generation Quality:\n")
            f.write(f"- BLEU-4: {results.bleu_4:.4f}\n")
            f.write(f"- ROUGE-L: {results.rouge_l:.4f}\n")
            f.write(f"- BERTScore F1: {results.bertscore_f1:.4f}\n")
            f.write(f"- Semantic Similarity: {results.semantic_similarity:.4f}\n\n")
            
            f.write(f"Medical Metrics:\n")
            f.write(f"- Clinical Accuracy: {results.clinical_accuracy:.4f}\n")
            f.write(f"- Safety Score: {results.safety_score:.4f}\n")
            f.write(f"- Factual Correctness: {results.factual_correctness:.4f}\n\n")
            
            f.write(f"Efficiency:\n")
            f.write(f"- Avg Inference Time: {results.inference_time_per_sample:.3f} seconds\n")
            f.write(f"- Unanswered Rate: {results.unanswered_rate:.4f}\n")
            
            if results.answer_distribution:
                f.write(f"\n Answer Distribution:\n")
                for choice, freq in sorted(results.answer_distribution.items()):
                    f.write(f"- Choice {choice}: {freq:.3f}\n")


if __name__ == "__main__":
    main()
