# %pip install trl

import subprocess
import sys

# Packages needed to install.
def install_requirements():
    packages = [
        "bitsandbytes",
        "accelerate",
        "peft",
        "trl",
        "datasets",
        "transformers",
        "torch",
        "wandb",
        "scikit-learn"
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"{package} already installed.")
        except ImportError:
            print(f"Installing {package}.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")

install_requirements()

import json
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import math

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetDict
import wandb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Logging setup for visibility.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model loading and configuration.
@dataclass
class ModelConfig:
    model_name: str = "skumar9/Llama-medx_v3.2"
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    device_map: str = "auto"
    max_memory: Optional[Dict] = None
    torch_dtype: str = "auto"
    low_cpu_mem_usage: bool = True

# LoRA fine-tuning.
@dataclass
class LoRAConfig:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    inference_mode: bool = False

# Training hyperparameters.
@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_steps: int = -1
    eval_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    evaluation_strategy: str = "steps"  
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    dataloader_num_workers: int = 0  # Reduced it to avoid multiprocessing issues.
    fp16: bool = True
    bf16: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    data_seed: int = 42

# Data processing.
@dataclass
class DataConfig:
    max_seq_length: int = 2048
    instruction_template: str = "### Human: {instruction}\n### Assistant:"
    response_template: str = " {output}"
    cot_template: str = "\n\nThinking step by step:\n{cot_reasoning}\n\nTherefore, "
    use_cot: bool = True
    pack_sequences: bool = False
    remove_unused_columns: bool = False

# For processing the medical data and preparing it.
class MedicalDataProcessor:

    def __init__(self, config: DataConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Setting up tokenizer special tokens.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Setting padding side for consistency.
        self.tokenizer.padding_side = "right"

    # Tokenizing one sample at a time.
    def tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = sample["text"]

        # Tokenizing the text with padding and truncation.
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",  # Pad to max_length for consistent tensor sizes.
            max_length=self.config.max_seq_length,
            return_tensors=None  # To return a simple dict instead of tensors.
        )

        # For causal LM, labels are the same as input_ids except for padding tokens.
        labels = tokenized["input_ids"].copy()

        # Set labels to -100 for padding tokens to ignore them in loss calculation.
        if "attention_mask" in tokenized:
            for i, mask in enumerate(tokenized["attention_mask"]):
                if mask == 0:
                    labels[i] = -100

        tokenized["labels"] = labels

        return tokenized

    def load_dataset(self, train_file: str, eval_file: Optional[str] = None) -> DatasetDict:
        logger.info(f"Loading training data from {train_file}")

        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        eval_data = None
        if eval_file and os.path.exists(eval_file):
            logger.info(f"Loading evaluation data from {eval_file}")
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        else:
            # Splitting training data for evaluation.
            split_idx = int(len(train_data) * 0.9)
            eval_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
            logger.info(f"Split training data: {len(train_data)} train, {len(eval_data)} eval")


        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)

        # Format the samples first to create a text field.
        train_dataset = train_dataset.map(
            self.format_sample,
            remove_columns=train_dataset.column_names,
            desc="Formatting training data"
        )

        eval_dataset = eval_dataset.map(
            self.format_sample,
            remove_columns=eval_dataset.column_names,
            desc="Formatting evaluation data"
        )

        # Tokenize.
        train_dataset = train_dataset.map(
            self.tokenize_sample,
            remove_columns=["text"],
            desc="Tokenizing training data"
        )

        eval_dataset = eval_dataset.map(
            self.tokenize_sample,
            remove_columns=["text"],
            desc="Tokenizing evaluation data"
        )

        return DatasetDict({
            "train": train_dataset,
            "eval": eval_dataset
        })

    # Formatting the sample.
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        cot_reasoning = sample.get("cot_reasoning", "") if self.config.use_cot else ""

        # Formatted text.
        human_part = self.config.instruction_template.format(instruction=instruction)

        if self.config.use_cot and cot_reasoning:
            # Including the CoT reasoning.
            cot_part = self.config.cot_template.format(cot_reasoning=cot_reasoning)
            response_part = self.config.response_template.format(output=output)
            full_text = human_part + cot_part + response_part
        else:
            # Standard format without CoT.
            response_part = self.config.response_template.format(output=output)
            full_text = human_part + response_part

        return {"text": full_text}

# Model trainer class that handles the entire training process.
class ModelTrainer:

    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        data_config: DataConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.data_config = data_config

        self.model = None
        self.tokenizer = None
        self.peft_model = None

    # Setting up quantization configuration.
    def setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if self.model_config.use_4bit:
            logger.info("Setting up 4-bit quantization.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.model_config.use_nested_quant,
                bnb_4bit_quant_type=self.model_config.bnb_4bit_quant_type,
            )
        elif self.model_config.use_8bit:
            logger.info("Setting up 8-bit quantization.")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None

    # Optimal device map based on GPU memory.
    def get_optimal_device_map(self):
        if not torch.cuda.is_available():
            return None

        # Checking the GPU memory.
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  
            print(f"Detected GPU memory: {gpu_memory:.1f} GB")

            if gpu_memory < 12:  
                print("Limited GPU memory detected, using simple device mapping.")
                return {"": 0}  
            else:
                return "auto"  # Use auto mapping for high-memory GPUs.
        except:
            return {"": 0}

    # Loading the base model and tokenizer.
    def load_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.model_config.model_name}")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                trust_remote_code=True,
                padding_side="right",
                use_fast=False
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            # Try with a fallback tokenizer
            logger.info("Trying fallback tokenizer.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium",  # Fallback tokenizer.
                trust_remote_code=True,
                padding_side="right",
                use_fast=False
            )

        # Setting up quantization with better error handling.
        quantization_config = None
        try:
            quantization_config = self.setup_quantization_config()
        except Exception as e:
            logger.warning(f"Quantization setup failed: {e}")
            logger.info("Continuing without quantization.")
            self.model_config.use_4bit = False
            self.model_config.use_8bit = False

        device_map = self.get_optimal_device_map()

        # Torch dtype.
        if self.model_config.torch_dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype = getattr(torch, self.model_config.torch_dtype)

        print(f"Using device map: {device_map}")
        print(f"Using quantization: {quantization_config is not None}")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=self.model_config.low_cpu_mem_usage,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
        except Exception as e:
            logger.warning(f"Failed to load model with flash attention: {e}")
            logger.info("Retrying without flash attention.")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=self.model_config.low_cpu_mem_usage
                )
            except Exception as e2:
                logger.warning(f"Failed with device mapping: {e2}")
                logger.info("Trying with simpler configuration.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )

        if quantization_config is not None:
            try:
                self.model = prepare_model_for_kbit_training(self.model)
            except Exception as e:
                logger.warning(f"Failed to prepare model for k-bit training: {e}")

        try:
            self.model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

        logger.info("Model and tokenizer loaded.")

    # Target modules for LoRA.
    def get_target_modules_for_model(self, model):
        model_name = self.model_config.model_name.lower()

        module_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                module_names.append(name.split('.')[-1])

        # Removes duplicates and common modules we don't want to target.
        unique_modules = list(set(module_names))
        exclude_modules = ['layernorm', 'ln', 'bias', 'embedding', 'lm_head', 'embed', 'norm']
        target_candidates = [m for m in unique_modules if not any(ex in m.lower() for ex in exclude_modules)]

        print(f"Available modules in model: {target_candidates}")

        # Define target modules based on the model type.
        if 'llama' in model_name or 'alpaca' in model_name:
            # Llama/Alpaca models.
            llama_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            available_targets = [t for t in llama_targets if t in target_candidates]
            if available_targets:
                return available_targets

        elif 'gpt' in model_name or 'dialo' in model_name:
            # GPT-based models.
            gpt_targets = ["c_attn", "c_proj", "c_fc", "attn", "mlp"]
            available_targets = [t for t in gpt_targets if t in target_candidates]
            if available_targets:
                return available_targets

        elif 'bert' in model_name:
            # BERT based models.
            bert_targets = ["query", "key", "value", "dense"]
            available_targets = [t for t in bert_targets if t in target_candidates]
            if available_targets:
                return available_targets

        # Fallback if no specific targets are found.
        linear_patterns = ["linear", "proj", "fc", "dense", "attn"]
        fallback_targets = []
        for pattern in linear_patterns:
            matches = [m for m in target_candidates if pattern in m.lower()]
            fallback_targets.extend(matches)

        if fallback_targets:
            # Takes the first few unique matches.
            return list(set(fallback_targets))[:4]

        # Last resort.
        if target_candidates:
            print(f"⚠️ Using fallback target modules: {target_candidates[:3]}")
            return target_candidates[:3]

        # If all else fails, it returns empty list and let PEFT handle it
        print("Could not determine target modules, using PEFT defaults.")
        return []

    # Setting up LoRA.
    def setup_lora(self):
        logger.info("Setting up LoRA configuration")

        # Skip LoRA if rank is 0 (for simple full fine-tuning)
        if self.lora_config.lora_r == 0:
            logger.info("LoRA disabled (rank=0), using full fine-tuning")
            self.peft_model = self.model
            return

        # Get appropriate target modules for this model
        target_modules = self.get_target_modules_for_model(self.model)

        if not target_modules:
            # If no target modules found, let PEFT auto-detect
            target_modules = None
            print("Using PEFT auto-detection for target modules.")
        else:
            print(f"Using target modules: {target_modules}")

        peft_config = LoraConfig(
            r=self.lora_config.lora_r,
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            bias=self.lora_config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=self.lora_config.inference_mode
        )

        # Apply LoRA.
        try:
            self.peft_model = get_peft_model(self.model, peft_config)
        except Exception as e:
            logger.warning(f"Failed to apply LoRA with auto-detected modules: {e}")
            logger.info("Trying with broader target module selection.")

            # Fallback to broader target modules.
            fallback_targets = ["linear", "Linear"]
            peft_config.target_modules = fallback_targets

            try:
                self.peft_model = get_peft_model(self.model, peft_config)
            except Exception as e2:
                logger.warning(f"Fallback also failed: {e2}")
                logger.info("Using model without LoRA.")
                self.peft_model = self.model
                return

        # Trainable parameters.
        try:
            self.peft_model.print_trainable_parameters()
        except:
            logger.info("Model ready for training.")

        logger.info("LoRA applied.")

    def setup_training_arguments(self, output_dir: str, dataset_size: int) -> TrainingArguments:
        if self.training_config.eval_steps == -1:
            self.training_config.eval_steps = max(50, dataset_size // (self.training_config.per_device_train_batch_size * 10))

        if self.training_config.save_steps == -1:
            self.training_config.save_steps = self.training_config.eval_steps * 2

        # Max steps if not provided.
        if self.training_config.max_steps == -1:
            steps_per_epoch = math.ceil(dataset_size / (
                self.training_config.per_device_train_batch_size *
                self.training_config.gradient_accumulation_steps
            ))
            self.training_config.max_steps = steps_per_epoch * self.training_config.num_train_epochs

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            warmup_ratio=self.training_config.warmup_ratio,
            max_steps=self.training_config.max_steps,
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            eval_strategy=self.training_config.evaluation_strategy,
            save_strategy=self.training_config.save_strategy,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            report_to=self.training_config.report_to,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            seed=self.training_config.seed,
            data_seed=self.training_config.data_seed,
            remove_unused_columns=self.data_config.remove_unused_columns,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            dataloader_pin_memory=True,
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        # In case that the predictions might be logits.
        if len(predictions.shape) > 2:
            predictions = predictions.argmax(-1)

        # Decode predictions and labels to text.
        decoded_preds = []
        decoded_labels = []

        for pred, label in zip(predictions, labels):
            # Remove 100 tokens from the labels.
            label_filtered = [token for token in label if token != -100]

            try:
                decoded_pred = self.tokenizer.decode(pred, skip_special_tokens=True)
                decoded_label = self.tokenizer.decode(label_filtered, skip_special_tokens=True)

                decoded_preds.append(decoded_pred)
                decoded_labels.append(decoded_label)
            except:
                decoded_preds.append("")
                decoded_labels.append("")

        # Metrics.
        pred_lengths = [len(pred.split()) for pred in decoded_preds if pred]
        label_lengths = [len(label.split()) for label in decoded_labels if label]

        return {
            "avg_pred_length": np.mean(pred_lengths) if pred_lengths else 0,
            "avg_label_length": np.mean(label_lengths) if label_lengths else 0,
            "length_ratio": (np.mean(pred_lengths) / np.mean(label_lengths)) if (pred_lengths and label_lengths and np.mean(label_lengths) > 0) else 0
        }

    def train(self, train_file: str, eval_file: Optional[str], output_dir: str):
        logger.info("Starting training process.")

        # Output directory.
        os.makedirs(output_dir, exist_ok=True)

        self.load_model_and_tokenizer()

        # Setting up LoRA.
        self.setup_lora()

        # Loading the data.
        data_processor = MedicalDataProcessor(self.data_config, self.tokenizer)
        dataset = data_processor.load_dataset(train_file, eval_file)

        logger.info(f"Training samples: {len(dataset['train'])}")
        logger.info(f"Evaluation samples: {len(dataset['eval'])}")

        # Training arguments.
        training_args = self.setup_training_arguments(output_dir, len(dataset['train']))

        # Setting up data collator.
        try:
            from transformers import default_data_collator
            data_collator = default_data_collator
            logger.info("Using default data collator with pre-padded sequences.")

        except Exception as e:
            logger.warning(f"Failed to setup default data collator: {e}")

            # Fallback to data collator for completion only LM.
            try:
                from transformers import DataCollatorForLanguageModeling
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False,  # We're doing causal LM, not masked LM.
                    pad_to_multiple_of=None,
                )
                logger.info("Using DataCollatorForLanguageModeling")
            except Exception as e2:
                logger.warning(f"Failed to setup DataCollatorForLanguageModeling: {e2}")

                # Final fallback to a simple data collator.
                def simple_data_collator(features):
                    import torch
                    batch = {}

                    # Getting all keys from the first feature.
                    keys = features[0].keys()

                    for key in keys:
                        batch[key] = torch.stack([torch.tensor(f[key]) for f in features])

                    return batch

                data_collator = simple_data_collator
                logger.info("Using custom simple data collator.")

        # Reducing the number of workers to avoid multiprocessing issues.
        training_args.dataloader_num_workers = 0 

        # Setup trainer with SFTTrainer or standard Trainer.
        try:
            trainer = SFTTrainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                dataset_text_field="text",
                packing=False,
            )

            # Set tokenizer manually if needed.
            if hasattr(trainer, 'tokenizer'):
                trainer.tokenizer = self.tokenizer
            else:
                setattr(trainer, 'tokenizer', self.tokenizer)

        except Exception as e:
            logger.warning(f"SFTTrainer failed: {e}")
            logger.info("Falling back to standard Trainer.")

            # Fallback to standard trainer.
            try:
                trainer = Trainer(
                    model=self.peft_model,
                    args=training_args,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["eval"],
                    data_collator=data_collator,
                    compute_metrics=self.compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                    processing_class=self.tokenizer,
                )
            except Exception as e2:
                logger.warning(f"Standard Trainer with processing_class failed: {e2}")
                trainer = Trainer(
                    model=self.peft_model,
                    args=training_args,
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["eval"],
                    data_collator=data_collator,
                    compute_metrics=self.compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                )
                # Set tokenizer as attribute.
                setattr(trainer, 'tokenizer', self.tokenizer)

        self.save_configs(output_dir)

        # Start training.
        logger.info("Starting training.")
        train_result = trainer.train()

        logger.info("Saving final model.")
        trainer.save_model(output_dir)

        # Training metrics.
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        logger.info("Running final evaluation.")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        # Save tokenizer.
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Training completed successfully. Model saved to {output_dir}")

        return train_result, eval_metrics

    def save_configs(self, output_dir: str):
        configs = {
            "model_config": self.model_config.__dict__,
            "lora_config": self.lora_config.__dict__,
            "training_config": self.training_config.__dict__,
            "data_config": self.data_config.__dict__
        }

        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=2)

        logger.info(f"Training configurations saved to {config_path}")

# Google Colab.
def upload_files():
    from google.colab import files
    print("Please select your training file to upload:")
    uploaded = files.upload()

    uploaded_files = {}
    for filename, content in uploaded.items():
        with open(filename, 'wb') as f:
            f.write(content)
        uploaded_files[filename] = filename
        print(f"Uploaded and saved: {filename}")

    return uploaded_files

def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive successful.")

def download_from_drive(file_path_in_drive, local_path):
    import shutil

    full_drive_path = f"/content/drive/MyDrive/{file_path_in_drive}"

    if os.path.exists(full_drive_path):
        shutil.copy2(full_drive_path, local_path)
        print(f"File downloaded from {full_drive_path} to {local_path}")
        return True
    else:
        print(f"File not found in Google Drive: {full_drive_path}")
        return False

# Function to get training file.
def get_training_file(method="upload"):
    if method == "upload":
        print("File Upload method.")
        uploaded_files = upload_files()

        train_file = None
        eval_file = None

        for filename in uploaded_files.keys():
            if "train" in filename.lower() and filename.endswith('.json'):
                train_file = filename
            elif "eval" in filename.lower() and filename.endswith('.json'):
                eval_file = filename

        if not train_file:
            json_files = [f for f in uploaded_files.keys() if f.endswith('.json')]
            if json_files:
                train_file = json_files[0]
                print(f"Using {train_file} as training file.")

        return train_file, eval_file

    elif method == "drive":
        print("Google Drive method.")
        mount_google_drive()

        train_path = input("Enter the path to your training file in Google Drive: ")
        eval_path = input("Enter the path to your evaluation file in Google Drive: ")

        local_train_file = "./train_data.json"
        local_eval_file = None

        if download_from_drive(train_path, local_train_file):
            if eval_path.strip():
                local_eval_file = "./eval_data.json"
                if not download_from_drive(eval_path, local_eval_file):
                    local_eval_file = None
            return local_train_file, local_eval_file
        else:
            raise FileNotFoundError(f"Could not download training file: {train_path}")

    else:
        if os.path.exists(method):
            return method, None
        else:
            raise FileNotFoundError(f"File not found: {method}")


def run_training(
    file_method: str = "upload",  
    train_file_drive_path: Optional[str] = None,  
    eval_file_drive_path: Optional[str] = None,   
    output_dir: str = "./fine_tuned_model",

    # Model configuration.
    model_name: str = "skumar9/Llama-medx_v3.2",
    use_4bit: bool = True,
    use_8bit: bool = False,

    # LoRA configuration.
    lora_r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,

    # Training configuration.
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    eval_steps: int = 100,
    save_steps: int = 200,

    # Data configuration.
    max_seq_length: int = 2048,
    use_cot: bool = True,

    # Experiment tracking.
    wandb_project: Optional[str] = None,
    run_name: Optional[str] = None
):
    """
    Run the complete fine-tuning process with flexible file input methods.

    Args:
        file_method: How to get training files:
            - "upload": Upload files directly via browser
            - "drive": Download from Google Drive
            - Or provide direct file path to existing file
        train_file_drive_path: Path to training file in Google Drive (if using "drive" method)
        eval_file_drive_path: Path to evaluation file in Google Drive (optional)
        output_dir: Local directory to save the fine-tuned model
        ... (other parameters as documented in the original code)
    """

    print("Starting Medical Chatbot Fine-tuning Process.")
    print("=" * 60)

    # Get training files based on method
    if file_method == "upload":
        train_file, eval_file = get_training_file("upload")
    elif file_method == "drive":
        if train_file_drive_path:
            # Use provided paths
            mount_google_drive()
            train_file = "./train_data.json"
            eval_file = None

            if not download_from_drive(train_file_drive_path, train_file):
                raise FileNotFoundError(f"Could not download training file: {train_file_drive_path}")

            if eval_file_drive_path:
                eval_file = "./eval_data.json"
                if not download_from_drive(eval_file_drive_path, eval_file):
                    eval_file = None
        else:
            # Interactive mode.
            train_file, eval_file = get_training_file("drive")
    else:
        train_file, eval_file = get_training_file(file_method)

    if not train_file:
        raise ValueError("No training file provided or found.")

    print(f"Training file: {train_file}")
    if eval_file:
        print(f"Evaluation file: {eval_file}")
    else:
        print("No evaluation file provided, we will split training data.")

    # Setup wandb.
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        if run_name:
            os.environ["WANDB_RUN_NAME"] = run_name

    # Create configurations.
    model_config = ModelConfig(
        model_name=model_name,
        use_4bit=use_4bit,
        use_8bit=use_8bit
    )

    lora_config = LoRAConfig(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    )

    training_config = TrainingConfig(
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        save_steps=save_steps,
        report_to=["wandb"] if wandb_project else ["tensorboard"]
    )

    data_config = DataConfig(
        max_seq_length=max_seq_length,
        use_cot=use_cot
    )

    # Initialize trainer.
    trainer = ModelTrainer(model_config, lora_config, training_config, data_config)

    try:
        # Start training.
        train_result, eval_metrics = trainer.train(
            train_file,
            eval_file,
            output_dir
        )

        # Final results.
        print("\n" + "="*60)
        print("Training completed.")
        print("="*60)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Final evaluation loss: {eval_metrics['eval_loss']:.4f}")
        print(f"Model saved to: {output_dir}")
        print("="*60)

        return train_result, eval_metrics

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# Example usage functions
def start_training_with_upload():
    print("Quick Start: Upload File Method.")
    print("This will prompt you to upload your train_with_cot.json file directly.")

    try:
        train_result, eval_metrics = run_training(
            file_method="upload",           
            epochs=1,                       
            batch_size=1,                   
            eval_steps=10,                  # Frequent evaluation for testing.
            save_steps=20,
            output_dir="./uploaded_model",
            use_4bit=False,                 # Disable quantization to avoid memory issues.
            use_8bit=False,
            max_seq_length=512,            # Shorter sequences for quick testing.
        )

        return train_result, eval_metrics

    except Exception as e:
        print(f"Training failed: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Try restarting the runtime.")
        print("2. Try the CPU-only version with smaller parameters.")
        print("3. Try a smaller model if memory is the issue.")

        # Offer different fallback options
        print("\nChoose a fallback option:")
        print("1. CPU-only mode (c)")
        print("2. Smaller model (s)")
        print("3. Exit (e)")

        response = input("Enter your choice (c/s/e): ").lower()
        if response == 'c':
            return start_training_cpu_mode()
        elif response == 's':
            return start_training_small_model()
        else:
            raise

# Smaller model training for memory efficiency.
def start_training_small_model():
    print("Starting with smaller model.")

    train_result, eval_metrics = run_training(
        file_method="upload",
        model_name="microsoft/DialoGPT-small",  # Smaller model for memory efficiency.
        epochs=2,
        batch_size=2,
        eval_steps=10,
        save_steps=20,
        output_dir="./small_model",
        use_4bit=False,
        use_8bit=False,
        max_seq_length=256,
        learning_rate=5e-5,
        lora_r=8,                      # Smaller LoRA rank.
        lora_alpha=8,                  # Adjusted alpha.
    )

    return train_result, eval_metrics

# Basic training mode with conservative settings.
def start_training_basic_mode():
    print("Starting basic training mode.")

    train_result, eval_metrics = run_training(
        file_method="upload",
        model_name="microsoft/DialoGPT-medium",  
        epochs=1,
        batch_size=1,
        eval_steps=5,
        save_steps=10,
        output_dir="./basic_model",
        use_4bit=False,
        use_8bit=False,
        max_seq_length=256,
        learning_rate=5e-6,            # Very conservative learning rate.
        lora_r=4,                      # Very small LoRA rank.
        lora_alpha=4,
    )

    return train_result, eval_metrics

# CPU-only training mode for fallback.
def start_training_cpu_mode():
    print("Starting CPU-only training mode.")

    train_result, eval_metrics = run_training(
        file_method="upload",
        epochs=1,
        batch_size=1,
        eval_steps=5,
        save_steps=10,
        output_dir="./cpu_model",
        use_4bit=False,
        use_8bit=False,
        max_seq_length=256,             # Very short for CPU. 
        learning_rate=5e-5,             # Higher LR for faster convergence. 
    )

    return train_result, eval_metrics

def start_training_with_drive():
    print("Google Drive Method.")
    print("This will mount Google Drive and ask for your file paths.")

    train_result, eval_metrics = run_training(
        file_method="drive",            
        epochs=2,                       
        batch_size=4,                   
        eval_steps=50,
        save_steps=100,
        output_dir="./drive_model"
    )

    return train_result, eval_metrics

# Function to start training with specific Google Drive paths.
def start_training_with_drive_paths(train_path, eval_path=None):
    print(f"Google Drive Method with provided paths.")
    print(f"Training file: {train_path}")
    if eval_path:
        print(f"Evaluation file: {eval_path}")

    train_result, eval_metrics = run_training(
        file_method="drive",
        train_file_drive_path=train_path,
        eval_file_drive_path=eval_path,
        epochs=2,                       # Reduced for testing.
        batch_size=4,                   
        eval_steps=50,
        save_steps=100,
        output_dir="./drive_model"
    )

    return train_result, eval_metrics

def example_custom_training():
    print("Custom Training configuration.")

    train_result, eval_metrics = run_training(
        file_method="upload",

        # Model settings.
        model_name="skumar9/Llama-medx_v3.2",
        use_4bit=True,

        # LoRA settings.
        lora_r=32,
        lora_alpha=16,
        lora_dropout=0.1,

        # Training settings.
        epochs=3,
        batch_size=2,                   
        learning_rate=1e-5,             # Lower learning rate.
        eval_steps=25,                  
        save_steps=50,

        # Data settings.
        max_seq_length=1024,
        use_cot=True,

        output_dir="./custom_model",

        # wandb_project="medical-chatbot",
        # run_name="custom-experiment-1"
    )

    return train_result, eval_metrics

print("Fine-tuning loaded.")
print("\n" + "="*60)
print("Multiple ways to upload the file:")
print("="*60)
print("\n Basic:")
print("   start_training_basic_mode()")
print(" Uses standard Trainer and works with any TRL version.")

print("\n Small model mode:")
print("   start_training_small_model()")
print("DialoGPT-small with adaptive LoRA.")

print("\n Upload file:")
print("   start_training_with_upload()")
print("Adapts to any model with smart error handling.")

print("\n CPU-only mode:")
print("   start_training_cpu_mode()")
print(" No GPU/quantization, works on any hardware.")

print("\n" + "="*60)
print("Quick Fix for SFTTrainer error:")
print("   The trainer now falls back to standard Trainer.")
print("   Try: start_training_basic_mode()")
print("="*60)

print("\n Trainer compatability features:")
print("- Auto-detects SFTTrainer vs standard Trainer compatibility.")
print("- Falls back gracefully if SFTTrainer parameters don't match.")
print("- Works with any version of transformers/TRL.")
print("- Handles data collator compatibility issues.")
print("- Maximum compatibility mode available.")

start_training_small_model()

# Loading the fine-tuned model.
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_finetuned_model():
    try:
        print("Loading fine-tuned model.")

        # Load tokenizer and model.
        tokenizer = AutoTokenizer.from_pretrained("./small_model")
        model = AutoModelForCausalLM.from_pretrained("./small_model")

        # Check if tokenizer has pad token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("Model loaded.")

        # Test with medical queries.
        test_prompts = [
            "What are the symptoms of diabetes?",
            "How is hypertension treated?",
            "What causes chest pain?",
            "Explain the difference between Type 1 and Type 2 diabetes."
        ]

        print("\n" + "="*60)
        print("Testing the fine-tuned medial model.")
        print("="*60)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n Test {i}: {prompt}")
            print("-" * 50)

            try:
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                # Generating the response.
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # Decoding the response.
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extracting the generated part.
                if prompt in response:
                    generated_text = response[len(prompt):].strip()
                else:
                    generated_text = response.strip()

                print(f"Response: {generated_text}")

            except Exception as e:
                print(f"Error generating response: {e}")
                continue

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\n Troubleshooting suggestions:")
        print("1. Make sure the model was saved correctly.")
        print("2. Check if the path './small_model' exists.")
        print("3. Try loading with different parameters.")

        # Try the alternative loading approach.
        try:
            print("\nTrying alternative loading method.")
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            model = PeftModel.from_pretrained(base_model, "./small_model")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

            print("Alternative loading successful.")

            # Test with one prompt.
            prompt = "What are the symptoms of diabetes?"
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=150, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")

        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")

def simple_test():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./small_model")
        model = AutoModelForCausalLM.from_pretrained("./small_model")

        # Simple test.
        prompt = "What are the symptoms of diabetes?"
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with simple parameters.
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

    except Exception as e:
        print(f"Error: {e}")

def check_model_files():
    import os

    model_path = "./small_model"
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        print(f"Files in {model_path}:")
        for file in files:
            print(f"  - {file}")
    else:
        print(f"Model path {model_path} does not exist.")

# Run the tests
print("Checking model files.")
check_model_files()

print("\nRunning simple test.")
simple_test()

print("\nRunning comprehensive test.")
test_finetuned_model()