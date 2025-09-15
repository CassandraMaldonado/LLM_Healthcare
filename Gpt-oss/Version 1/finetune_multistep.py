# Fine-tunning GPT-OSS-20B model sequentially on PubMedQA, MedMCQA and CoT RadQA datasets.

conda create -n medical-qa python=3.9
conda activate medical-qa

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.36.0
!pip install peft==0.7.1
!pip install datasets==2.14.0
!pip install bitsandbytes==0.41.3
!pip install accelerate==0.25.0
!pip install wandb
!pip install jsonlines

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import wandb
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/gpt-oss-20b")
    trust_remote_code: bool = field(default=True)
    use_8bit: bool = field(default=True)
    

@dataclass
class DataArguments:
    pubmedqa_path: str = field(default="pubmedqa_train.jsonl")
    medmcqa_path: str = field(default="medmcqa_train.jsonl")
    medqa_path: str = field(default="medqa_train.jsonl")
    max_seq_length: int = field(default=2048)
    

@dataclass
class LoraArguments:
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class TrainingStageArguments:
    output_dir: str = field(default="./checkpoints")
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    eval_steps: int = field(default=200)
    save_steps: int = field(default=500)
    logging_steps: int = field(default=50)
    warmup_ratio: float = field(default=0.03)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.001)
    fp16: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)


class InstructionDataset:
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(data)} examples from {data_path}.")
        return data

# Instruction, input and output into a single string.   
    def format_instruction(self, example: Dict) -> str:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt

# Tokenizing the formatted text with padding and truncation.
    def tokenize_function(self, examples):
        formatted_texts = [self.format_instruction(ex) for ex in examples]
        
        model_inputs = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        # Setting labels for language modeling.
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs

# Converting to HuggingFace Dataset.  
    def get_dataset(self) -> Dataset:
        dataset = Dataset.from_list(self.data)
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenize_function([examples]),
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )
        return tokenized_dataset


# 8-bit quantization.
def setup_quantization_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )


# Lora.
def setup_lora_config(lora_args: LoraArguments):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias="none",
    )


def load_model_and_tokenizer(model_args: ModelArguments, lora_config: LoraConfig):

    quantization_config = setup_quantization_config() if model_args.use_8bit else None
    
    # Loading tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # Adding pad token if its missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Loading the model.
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )


class CustomTrainer(Trainer):
    
    def log(self, logs: Dict[str, float]) -> None:
        super().log(logs)
        
        # Log metrics to wandb.
        if wandb.run is not None:
            wandb_logs = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    wandb_logs[key] = value
            
            if wandb_logs:
                wandb.log(wandb_logs, step=self.state.global_step)


def train_stage(
    stage_name: str,
    data_path: str,
    model,
    tokenizer,
    training_args: TrainingArguments,
    data_args: DataArguments,
    resume_from_checkpoint: Optional[str] = None,
):
    """Train a single stage."""
    logger.info(f"Starting training stage: {stage_name}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output directory: {training_args.output_dir}")
    
    # Loading the dataset.
    instruction_dataset = InstructionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
    )
    train_dataset = instruction_dataset.get_dataset()
    

    data_collator = create_data_collator(tokenizer)
    
    # Trainer.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    

    if resume_from_checkpoint:
        if os.path.isdir(resume_from_checkpoint):
            logger.info(f"Resuming training from {resume_from_checkpoint}.")
        else:
            logger.warning(f"Checkpoint directory {resume_from_checkpoint} not found.")
            resume_from_checkpoint = None
    
    # Training.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Checkpoint.
    trainer.save_model()
    trainer.save_state()
    
    logger.info(f"Completed training stage: {stage_name}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Multi-stage medical QA fine-tuning")
    
    parser.add_argument("--model_name_or_path", default="mistralai/gpt-oss-20b", help="Model path")
    parser.add_argument("--pubmedqa_path", default="pubmedqa_train.jsonl", help="PubMedQA dataset path")
    parser.add_argument("--medmcqa_path", default="medmcqa_train.jsonl", help="MedMCQA dataset path")
    parser.add_argument("--medqa_path", default="medqa_train.jsonl", help="MedQA dataset path")
    parser.add_argument("--output_dir", default="./checkpoints", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs per stage")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--stage", type=str, choices=["all", "1", "2", "3"], default="all", help="Which stage to run")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--wandb_project", type=str, default="medical-qa-curriculum", help="W&B project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_8bit", action="store_true", default=True, help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    # Seed.
    set_seed(args.seed)
    
    wandb.init(
        project=args.wandb_project,
        name=f"curriculum-finetune-{args.stage}",
        config=args,
    )
    

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_8bit=args.use_8bit,
    )
    
    data_args = DataArguments(
        pubmedqa_path=args.pubmedqa_path,
        medmcqa_path=args.medmcqa_path,
        medqa_path=args.medqa_path,
        max_seq_length=args.max_seq_length,
    )
    
    lora_args = LoraArguments(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    

    lora_config = setup_lora_config(lora_args)
    
    # Training stages.
    stages = [
        {
            "name": "pubmedqa",
            "data_path": args.pubmedqa_path,
            "output_dir": os.path.join(args.output_dir, "pubmedqa"),
        },
        {
            "name": "medmcqa", 
            "data_path": args.medmcqa_path,
            "output_dir": os.path.join(args.output_dir, "medmcqa"),
        },
        {
            "name": "medqa",
            "data_path": args.medqa_path,
            "output_dir": os.path.join(args.output_dir, "medqa"),
        },
    ]
    
    # Which stages to run.
    if args.stage == "all":
        stages_to_run = [0, 1, 2]
    else:
        stages_to_run = [int(args.stage) - 1]
    
    model = None
    tokenizer = None
    
    for stage_idx in stages_to_run:
        stage = stages[stage_idx]
        stage_name = stage["name"]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"STAGE {stage_idx + 1}: {stage_name.upper()}")
        logger.info(f"{'='*50}")
        
        # Base model.
        if model is None:
            if stage_idx > 0 and args.stage == "all":
                prev_stage = stages[stage_idx - 1]
                model_path = prev_stage["output_dir"]
                logger.info(f"Loading model from previous stage: {model_path}")
                
                # Tokenizer.
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # LoRA model.
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    quantization_config=setup_quantization_config() if model_args.use_8bit else None,
                    device_map="auto",
                    trust_remote_code=model_args.trust_remote_code,
                    torch_dtype=torch.float16,
                )
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model, tokenizer = load_model_and_tokenizer(model_args, lora_config)
        
        # Setting up training arguments.
        training_args = TrainingArguments(
            output_dir=stage["output_dir"],
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_steps=200,
            save_steps=500,
            logging_steps=50,
            warmup_ratio=0.03,
            learning_rate=args.learning_rate,
            weight_decay=0.001,
            fp16=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            run_name=f"{stage_name}-stage",
            report_to="wandb",
            save_total_limit=3,
            load_best_model_at_end=False,
        )
        
        # Train stage.
        trainer = train_stage(
            stage_name=stage_name,
            data_path=stage["data_path"],
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            data_args=data_args,
            resume_from_checkpoint=args.resume_from_checkpoint if stage_idx == stages_to_run[0] else None,
        )
        
        logger.info(f"Stage {stage_idx + 1} ({stage_name}) completed.")
        
        model = trainer.model
    
    wandb.finish()
    logger.info("All training stages completed.")


if __name__ == "__main__":
    main()
