# GPT-OSS In-Context Learning

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
import torch
import random
import gc


@dataclass
class POCConfig:
    model_path: str
    backend: str = "unsloth"
    stages: List[str] = field(default_factory=lambda: ["stage1.jsonl", "stage2.jsonl"])
    subset_size: int = 10000
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 150
    batch_size: int = 4
    num_icl_examples: int = 30
    max_prompt_tokens: int = 14000
    output_dir: str = "./poc_results"
    max_gpu_hours: float = 40.0
    save_every: int = 500
    seed: int = 42
    quantization: str = "4bit"  # 4-bit quantization
    max_seq_length: int = 16384  # 16K context
    
    def __post_init__(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)


class Timer:
    """Track elapsed time and estimate remaining time."""
    
    def __init__(self, max_hours: float):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600
        self.samples_processed = 0
        self.stage_start = None
        
    def start_stage(self):
        """Mark the start of a new stage."""
        self.stage_start = time.time()
        
    def elapsed_hours(self) -> float:
        """Get total elapsed hours."""
        return (time.time() - self.start_time) / 3600
    
    def stage_elapsed_minutes(self) -> float:
        """Get minutes elapsed in current stage."""
        if self.stage_start:
            return (time.time() - self.stage_start) / 60
        return 0
    
    def should_stop(self) -> bool:
        """Check if we've exceeded time budget."""
        return time.time() - self.start_time > self.max_seconds
    
    def estimate_time_per_sample(self, samples_done: int, stage_start_time: float) -> float:
        """Estimate seconds per sample for current stage."""
        if samples_done == 0:
            return 0
        elapsed = time.time() - stage_start_time
        return elapsed / samples_done
    
    def format_eta(self, samples_remaining: int, time_per_sample: float) -> str:
        """Format estimated time remaining."""
        seconds = samples_remaining * time_per_sample
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class MemoryMonitor:
    """Monitor GPU memory usage."""
    
    @staticmethod
    def get_gpu_memory_mb() -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    @staticmethod
    def clear_cache():
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def print_memory_stats():
        """Print current memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class DataLoader:
    """Efficient dataset loading with subset support."""
    
    @staticmethod
    def load_stage(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load stage file with optional limit.
        
        Args:
            file_path: Path to JSONL file
            max_samples: Maximum samples to load (None = all)
            
        Returns:
            List of sample dictionaries
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    sample = json.loads(line)
                    if DataLoader._validate(sample):
                        data.append(sample)
        return data
    
    @staticmethod
    def _validate(sample: Dict) -> bool:
        """Check if sample has required fields."""
        return all(k in sample for k in ['input', 'output'])
    
    @staticmethod
    def count_lines(file_path: str) -> int:
        """Count total lines in file."""
        with open(file_path, 'r') as f:
            return sum(1 for line in f if line.strip())


class PromptBuilder:
    """Build optimized ICL prompts."""
    
    @staticmethod
    def build_prompt(
        examples: List[Dict],
        query: str,
        max_tokens: int = 14000
    ) -> Tuple[str, int]:
        """
        Build ICL prompt with dynamic fitting.
        
        Args:
            examples: ICL example pool
            query: Current question
            max_tokens: Maximum allowed tokens
            
        Returns:
            (prompt_string, num_examples_used)
        """
        instruction = (
            "You are a medical expert. Study these examples carefully, "
            "then answer the new question with only the treatment/diagnosis name.\n"
        )
        
        # Estimate base tokens
        base = f"{instruction}\n### New Question\n{query}\n\n### Answer\n"
        base_tokens = len(base.split()) * 1.3  # Rough estimate
        
        # Fit examples
        available_tokens = max_tokens - base_tokens
        fitted_examples = []
        current_tokens = 0
        
        for ex in examples:
            ex_text = f"Q: {ex['input']}\nA: {ex['output']}\n\n"
            ex_tokens = len(ex_text.split()) * 1.3
            
            if current_tokens + ex_tokens > available_tokens:
                break
                
            fitted_examples.append(ex)
            current_tokens += ex_tokens
        
        # Build final prompt
        if fitted_examples:
            examples_text = "\n".join([
                f"Q: {ex['input']}\nA: {ex['output']}"
                for ex in fitted_examples
            ])
            prompt = f"{instruction}\n### Examples\n{examples_text}\n\n{base}"
        else:
            prompt = base
            
        return prompt, len(fitted_examples)
    
    @staticmethod
    def select_examples(
        pool: List[Dict],
        n: int,
        current_sample: Dict,
        seed: int
    ) -> List[Dict]:
        """Randomly select n examples, excluding current sample."""
        random.seed(seed)
        candidates = [s for s in pool if s != current_sample]
        return random.sample(candidates, min(n, len(candidates)))


class UnslothModel:
    """Unsloth model wrapper with 4-bit quantization."""
    
    def __init__(self, config: POCConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load()
    
    def _load(self):
        """Load model with 4-bit quantization."""
        print(f"\n🔧 Loading model: {self.config.model_path}")
        print(f"   Quantization: {self.config.quantization}")
        print(f"   Max sequence length: {self.config.max_seq_length}")
        
        try:
            from unsloth import FastLanguageModel
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_path,
                max_seq_length=self.config.max_seq_length,
                dtype=None,
                load_in_4bit=(self.config.quantization == "4bit"),
            )
            
            FastLanguageModel.for_inference(self.model)
            
            print("✅ Model loaded successfully")
            MemoryMonitor.print_memory_stats()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"\n⚠️  Generation error: {e}")
            return ""
    
    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        MemoryMonitor.clear_cache()


class POCEvaluator:
    """Main evaluator for proof-of-concept experiment."""
    
    def __init__(self, config: POCConfig):
        self.config = config
        self.timer = Timer(config.max_gpu_hours)
        self.model = UnslothModel(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def evaluate_stage(
        self,
        stage_file: str,
        stage_num: int,
        max_samples: Optional[int] = None
    ) -> Dict:
        """
        Evaluate a single stage.
        
        Args:
            stage_file: Path to stage JSONL
            stage_num: Stage number (1 or 2)
            max_samples: Limit samples (for stage 2)
            
        Returns:
            Stage results dictionary
        """
        print(f"\n{'='*70}")
        print(f"📊 STAGE {stage_num}: {stage_file}")
        print(f"{'='*70}")
        
        # Check time budget
        if self.timer.should_stop():
            print("⏰ Time budget exceeded! Stopping evaluation.")
            return {}
        
        self.timer.start_stage()
        stage_start_time = time.time()
        
        # Count total samples
        total_in_file = DataLoader.count_lines(stage_file)
        print(f"   Total samples in file: {total_in_file:,}")
        
        if max_samples:
            print(f"   Loading subset: {max_samples:,} samples")
        else:
            print(f"   Loading all samples")
        
        # Load data
        data = DataLoader.load_stage(stage_file, max_samples)
        print(f"   Loaded: {len(data):,} valid samples")
        
        # Setup
        results = []
        correct = 0
        stage_name = f"stage{stage_num}"
        
        # Progress tracking
        pbar = tqdm(total=len(data), desc=f"Stage {stage_num}", ncols=100)
        
        for i, sample in enumerate(data):
            # Time budget check
            if self.timer.should_stop():
                print(f"\n⏰ Reached {self.config.max_gpu_hours}h time limit. Stopping.")
                break
            
            # Select ICL examples
            examples = PromptBuilder.select_examples(
                pool=data,
                n=self.config.num_icl_examples,
                current_sample=sample,
                seed=self.config.seed + i
            )
            
            # Build prompt
            prompt, num_used = PromptBuilder.build_prompt(
                examples=examples,
                query=sample['input'],
                max_tokens=self.config.max_prompt_tokens
            )
            
            # Generate
            prediction = self.model.generate(prompt)
            
            # Evaluate
            reference = sample['output'].strip()
            is_correct = self._check_match(prediction, reference)
            
            if is_correct:
                correct += 1
            
            # Store result
            results.append({
                'stage': stage_name,
                'sample_id': i,
                'question': sample['input'][:150] + "...",
                'prediction': prediction,
                'reference': reference,
                'correct': int(is_correct),
                'icl_examples_used': num_used
            })
            
            # Update progress
            pbar.update(1)
            
            # Time estimate
            if (i + 1) % 100 == 0:
                time_per_sample = self.timer.estimate_time_per_sample(
                    i + 1, stage_start_time
                )
                eta = self.timer.format_eta(len(data) - i - 1, time_per_sample)
                acc = (correct / (i + 1)) * 100
                pbar.set_postfix({
                    'acc': f'{acc:.1f}%',
                    'eta': eta,
                    'ex': num_used
                })
            
            # Periodic save
            if (i + 1) % self.config.save_every == 0:
                self._save_intermediate(results, stage_name, i + 1)
                MemoryMonitor.clear_cache()
        
        pbar.close()
        
        # Calculate metrics
        accuracy = (correct / len(results)) * 100 if results else 0
        elapsed_min = self.timer.stage_elapsed_minutes()
        
        # Save final results
        self._save_results(results, stage_name)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📈 STAGE {stage_num} SUMMARY")
        print(f"{'='*70}")
        print(f"   Samples processed: {len(results):,}")
        print(f"   Correct: {correct:,}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Runtime: {elapsed_min:.1f} minutes")
        print(f"   Avg time/sample: {(elapsed_min * 60 / len(results)):.2f}s")
        print(f"   Results: {self.config.output_dir}/{stage_name}_results.json")
        print(f"{'='*70}")
        
        return {
            'stage': stage_name,
            'samples': len(results),
            'correct': correct,
            'accuracy': accuracy,
            'runtime_minutes': elapsed_min
        }
    
    @staticmethod
    def _check_match(prediction: str, reference: str) -> bool:
        """Check if prediction matches reference (flexible matching)."""
        pred = prediction.lower().strip()
        ref = reference.lower().strip()
        
        # Exact match or containment
        return ref in pred or pred == ref
    
    def _save_intermediate(self, results: List[Dict], stage: str, count: int):
        """Save intermediate checkpoint."""
        path = Path(self.config.output_dir) / f"{stage}_checkpoint_{count}.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_results(self, results: List[Dict], stage: str):
        """Save final stage results."""
        base_path = Path(self.config.output_dir) / stage
        
        # JSON
        with open(f"{base_path}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # CSV
        df = pd.DataFrame(results)
        df.to_csv(f"{base_path}_results.csv", index=False)
        
        # Summary
        summary = {
            'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'accuracy': (sum(r['correct'] for r in results) / len(results) * 100) if results else 0,
            'avg_icl_examples': sum(r['icl_examples_used'] for r in results) / len(results) if results else 0
        }
        with open(f"{base_path}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run(self):
        """Run the full proof-of-concept experiment."""
        print("\n" + "="*70)
        print("🚀 GPT-OSS IN-CONTEXT LEARNING - PROOF OF CONCEPT")
        print("="*70)
        print(f"   Model: {self.config.model_path}")
        print(f"   Quantization: {self.config.quantization}")
        print(f"   Context window: {self.config.max_seq_length:,} tokens")
        print(f"   ICL examples: {self.config.num_icl_examples}")
        print(f"   Max GPU hours: {self.config.max_gpu_hours}")
        print(f"   Output: {self.config.output_dir}")
        print("="*70)
        
        MemoryMonitor.print_memory_stats()
        
        all_results = []
        
        # Stage 1: Full dataset
        if len(self.config.stages) >= 1 and Path(self.config.stages[0]).exists():
            result = self.evaluate_stage(
                stage_file=self.config.stages[0],
                stage_num=1,
                max_samples=None  # Full dataset
            )
            if result:
                all_results.append(result)
        
        # Stage 2: Subset
        if len(self.config.stages) >= 2 and Path(self.config.stages[1]).exists():
            if not self.timer.should_stop():
                result = self.evaluate_stage(
                    stage_file=self.config.stages[1],
                    stage_num=2,
                    max_samples=self.config.subset_size
                )
                if result:
                    all_results.append(result)
        
        # Final summary
        self._print_final_summary(all_results)
        
        # Cleanup
        self.model.cleanup()
        
        print("\n✅ Proof-of-concept complete!\n")
    
    def _print_final_summary(self, results: List[Dict]):
        """Print overall experiment summary."""
        print(f"\n{'='*70}")
        print("🎯 FINAL SUMMARY")
        print(f"{'='*70}")
        
        total_samples = sum(r['samples'] for r in results)
        total_correct = sum(r['correct'] for r in results)
        total_time = self.timer.elapsed_hours()
        
        print(f"   Stages completed: {len(results)}")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total correct: {total_correct:,}")
        print(f"   Overall accuracy: {(total_correct/total_samples*100):.2f}%")
        print(f"   Total runtime: {total_time:.2f} hours")
        print(f"   Avg time/sample: {(total_time*3600/total_samples):.2f}s")
        
        for r in results:
            print(f"\n   {r['stage']}: {r['accuracy']:.2f}% ({r['samples']:,} samples, {r['runtime_minutes']:.1f}min)")
        
        print(f"{'='*70}")
        
        # Save overall summary
        summary_path = Path(self.config.output_dir) / "poc_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'config': {
                    'model': self.config.model_path,
                    'quantization': self.config.quantization,
                    'num_icl_examples': self.config.num_icl_examples,
                    'max_gpu_hours': self.config.max_gpu_hours
                },
                'results': results,
                'overall': {
                    'total_samples': total_samples,
                    'total_correct': total_correct,
                    'accuracy': (total_correct/total_samples*100) if total_samples > 0 else 0,
                    'runtime_hours': total_time
                }
            }, f, indent=2)
        
        print(f"\n📁 Summary saved: {summary_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPT-OSS ICL Proof-of-Concept (Budget-Optimized for A100 40GB)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to GPT-OSS model"
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="unsloth",
        choices=["unsloth"],
        help="Backend (only Unsloth supported in POC)"
    )
    
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["stage1.jsonl", "stage2.jsonl"],
        help="Stage files to process"
    )
    
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Subset size for stage 2 (default: 10000)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling (default: 0.9)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Max tokens to generate (default: 150)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4, not used in sequential mode)"
    )
    
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=30,
        help="ICL examples per prompt (default: 30)"
    )
    
    parser.add_argument(
        "--max_gpu_hours",
        type=float,
        default=40.0,
        help="Maximum GPU hours before stopping (default: 40.0)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./poc_results",
        help="Output directory (default: ./poc_results)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = POCConfig(
        model_path=args.model_path,
        backend=args.backend,
        stages=args.stages,
        subset_size=args.subset_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        num_icl_examples=args.num_icl_examples,
        max_gpu_hours=args.max_gpu_hours,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Run experiment
    evaluator = POCEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()
