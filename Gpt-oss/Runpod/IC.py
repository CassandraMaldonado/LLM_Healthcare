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
    """Configuration for proof-of-concept ICL experiment."""
    model_path: str
    stages: List[str] = field(default_factory=lambda: ["data/stage1.jsonl", "data/stage2.jsonl"])
    subset_size: int = 10000
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 150
    num_icl_examples: int = 30
    max_prompt_tokens: int = 14000
    output_dir: str = "./poc_results"
    max_gpu_hours: float = 40.0
    save_every: int = 500
    seed: int = 42
    
    def __post_init__(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)


class Timer:
    """Track elapsed time and estimate remaining time."""
    
    def __init__(self, max_hours: float):
        self.start_time = time.time()
        self.max_seconds = max_hours * 3600
        self.stage_start = None
        
    def start_stage(self):
        self.stage_start = time.time()
        
    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600
    
    def stage_elapsed_minutes(self) -> float:
        if self.stage_start:
            return (time.time() - self.stage_start) / 60
        return 0
    
    def should_stop(self) -> bool:
        return time.time() - self.start_time > self.max_seconds


class MemoryMonitor:
    """Monitor GPU memory usage."""
    
    @staticmethod
    def get_gpu_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def print_memory_stats():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class DataLoader:
    """Efficient dataset loading with subset support."""
    
    @staticmethod
    def load_stage(file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
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
        return all(k in sample for k in ['input', 'output'])
    
    @staticmethod
    def count_lines(file_path: str) -> int:
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
        instruction = (
            "You are a medical expert. Study these examples carefully, "
            "then answer the new question with only the treatment/diagnosis name.\n"
        )
        
        base = f"{instruction}\n### New Question\n{query}\n\n### Answer\n"
        base_tokens = len(base.split()) * 1.3
        
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
        random.seed(seed)
        candidates = [s for s in pool if s != current_sample]
        return random.sample(candidates, min(n, len(candidates)))


class StandardModel:
    """Standard transformers model wrapper."""
    
    def __init__(self, config: POCConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load()
    
    def _load(self):
        print(f"\nLoading model: {self.config.model_path}")
        print(f"   Using: Standard Transformers (float16)")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print("Model loaded successfully")
            MemoryMonitor.print_memory_stats()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384
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
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"\nGeneration error: {e}")
            return ""
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        MemoryMonitor.clear_cache()


class POCEvaluator:
    """Main evaluator for proof-of-concept experiment."""
    
    def __init__(self, config: POCConfig):
        self.config = config
        self.timer = Timer(config.max_gpu_hours)
        self.model = StandardModel(config)
        
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_stage(
        self,
        stage_file: str,
        stage_num: int,
        max_samples: Optional[int] = None
    ) -> Dict:
        print(f"\n{'='*70}")
        print(f"STAGE {stage_num}: {stage_file}")
        print(f"{'='*70}")
        
        if self.timer.should_stop():
            print("Time budget exceeded! Stopping evaluation.")
            return {}
        
        self.timer.start_stage()
        
        total_in_file = DataLoader.count_lines(stage_file)
        print(f"   Total samples in file: {total_in_file:,}")
        
        if max_samples:
            print(f"   Loading subset: {max_samples:,} samples")
        
        data = DataLoader.load_stage(stage_file, max_samples)
        print(f"   Loaded: {len(data):,} valid samples")
        
        results = []
        correct = 0
        stage_name = f"stage{stage_num}"
        total_examples_used = 0
        
        pbar = tqdm(total=len(data), desc=f"Stage {stage_num}", ncols=100)
        
        for i, sample in enumerate(data):
            if self.timer.should_stop():
                print(f"\nReached {self.config.max_gpu_hours}h time limit. Stopping.")
                break
            
            examples = PromptBuilder.select_examples(
                pool=data,
                n=self.config.num_icl_examples,
                current_sample=sample,
                seed=self.config.seed + i
            )
            
            prompt, num_used = PromptBuilder.build_prompt(
                examples=examples,
                query=sample['input'],
                max_tokens=self.config.max_prompt_tokens
            )
            
            prediction = self.model.generate(prompt)
            
            reference = sample['output'].strip()
            is_correct = self._check_match(prediction, reference)
            
            if is_correct:
                correct += 1
            
            total_examples_used += num_used
            
            results.append({
                'stage': stage_name,
                'sample_id': i,
                'question': sample['input'][:150] + "...",
                'prediction': prediction,
                'reference': reference,
                'correct': int(is_correct),
                'icl_examples_used': num_used
            })
            
            pbar.update(1)
            
            if (i + 1) % 100 == 0:
                acc = (correct / (i + 1)) * 100
                pbar.set_postfix({'acc': f'{acc:.1f}%', 'ex': num_used})
            
            if (i + 1) % self.config.save_every == 0:
                self._save_intermediate(results, stage_name, i + 1)
                MemoryMonitor.clear_cache()
        
        pbar.close()
        
        accuracy = (correct / len(results)) * 100 if results else 0
        avg_examples = total_examples_used / len(results) if results else 0
        elapsed_min = self.timer.stage_elapsed_minutes()
        
        self._save_results(results, stage_name)
        
        print(f"\n{'='*70}")
        print(f"STAGE {stage_num} SUMMARY")
        print(f"{'='*70}")
        print(f"   Samples processed: {len(results):,}")
        print(f"   Correct: {correct:,}")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Avg ICL examples used: {avg_examples:.1f}")
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
        pred = prediction.lower().strip()
        ref = reference.lower().strip()
        return ref in pred or pred == ref
    
    def _save_intermediate(self, results: List[Dict], stage: str, count: int):
        path = Path(self.config.output_dir) / f"{stage}_checkpoint_{count}.json"
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_results(self, results: List[Dict], stage: str):
        base_path = Path(self.config.output_dir) / stage
        
        with open(f"{base_path}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        df = pd.DataFrame(results)
        df.to_csv(f"{base_path}_results.csv", index=False)
        
        summary = {
            'total': len(results),
            'correct': sum(r['correct'] for r in results),
            'accuracy': (sum(r['correct'] for r in results) / len(results) * 100) if results else 0,
            'avg_icl_examples': sum(r['icl_examples_used'] for r in results) / len(results) if results else 0
        }
        with open(f"{base_path}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run(self):
        print("\n" + "="*70)
        print("GPT-OSS IN-CONTEXT LEARNING - PROOF OF CONCEPT")
        print("="*70)
        print(f"   Model: {self.config.model_path}")
        print(f"   Backend: Standard Transformers (float16)")
        print(f"   ICL examples: {self.config.num_icl_examples}")
        print(f"   Max GPU hours: {self.config.max_gpu_hours}")
        print(f"   Output: {self.config.output_dir}")
        print("="*70)
        
        MemoryMonitor.print_memory_stats()
        
        all_results = []
        
        if len(self.config.stages) >= 1 and Path(self.config.stages[0]).exists():
            result = self.evaluate_stage(
                stage_file=self.config.stages[0],
                stage_num=1,
                max_samples=None
            )
            if result:
                all_results.append(result)
        
        if len(self.config.stages) >= 2 and Path(self.config.stages[1]).exists():
            if not self.timer.should_stop():
                result = self.evaluate_stage(
                    stage_file=self.config.stages[1],
                    stage_num=2,
                    max_samples=self.config.subset_size
                )
                if result:
                    all_results.append(result)
        
        self._print_final_summary(all_results)
        self.model.cleanup()
        
        print("\nProof-of-concept complete!\n")
    
    def _print_final_summary(self, results: List[Dict]):
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        
        total_samples = sum(r['samples'] for r in results)
        total_correct = sum(r['correct'] for r in results)
        total_time = self.timer.elapsed_hours()
        
        print(f"   Stages completed: {len(results)}")
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total correct: {total_correct:,}")
        print(f"   Overall accuracy: {(total_correct/total_samples*100):.2f}%")
        print(f"   Total runtime: {total_time:.2f} hours")
        
        for r in results:
            print(f"\n   {r['stage']}: {r['accuracy']:.2f}% ({r['samples']:,} samples)")
        
        print(f"{'='*70}")
        
        summary_path = Path(self.config.output_dir) / "poc_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'results': results,
                'overall': {
                    'total_samples': total_samples,
                    'total_correct': total_correct,
                    'accuracy': (total_correct/total_samples*100) if total_samples > 0 else 0,
                    'runtime_hours': total_time
                }
            }, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--stages", nargs="+", default=["data/stage1.jsonl", "data/stage2.jsonl"])
    parser.add_argument("--subset_size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--num_icl_examples", type=int, default=30)
    parser.add_argument("--max_gpu_hours", type=float, default=40.0)
    parser.add_argument("--output_dir", type=str, default="./poc_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = POCConfig(
        model_path=args.model_path,
        stages=args.stages,
        subset_size=args.subset_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_icl_examples=args.num_icl_examples,
        max_gpu_hours=args.max_gpu_hours,
        output_dir=args.output_dir,
        seed=args.seed
    )
    evaluator = POCEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    main()