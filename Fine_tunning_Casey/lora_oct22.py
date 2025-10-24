from huggingface_hub import login
login(".....")

import os, gc, json, random
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "lora-med-eval-gpuonly"
USE_CHAT   = True

# Micro-batch + grad accumulation
BATCH_SIZE_TRAIN   = 1
BATCH_SIZE_EVAL    = 1
GRADIENT_ACC_STEPS = 8

# Context length (lower if OOM)
MAX_SEQ_LEN        = 384

# Training hyperparams
LR                 = 1e-4
NUM_EPOCHS         = 3
RANDOM_SEED        = 42
WEIGHT_DECAY       = 0.0
WARMUP_RATIO       = 0.03
LOGGING_STEPS      = 10
SAVE_TOTAL_LIMIT   = 1
EVAL_STRATEGY      = "no"    # keep "no" to reduce memory usage during training

# LoRA config (use ULTRA_LOW_MEM=True for even lower memory)
ULTRA_LOW_MEM      = False   # if True → r=8 and target=["q_proj","v_proj"]
LORA_R             = 8 if ULTRA_LOW_MEM else 16
LORA_ALPHA         = 16 if ULTRA_LOW_MEM else 32
LORA_DROPOUT       = 0.0
LORA_TARGET_MODULES= ["q_proj","v_proj"] if ULTRA_LOW_MEM else ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

# Environment knobs (safer CUDA allocator)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

# ------------------------
# Simple GPU memory guard
# ------------------------
def bytes_to_gib(x): return x / (1024**3)

def assert_free_mem(min_gib_required: float):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but GPU-only training was requested.")
    free, total = torch.cuda.mem_get_info()
    free_gib = bytes_to_gib(free)
    if free_gib < min_gib_required:
        raise RuntimeError(
            f"Not enough free GPU memory: {free_gib:.2f} GiB free, require ≥ {min_gib_required:.2f} GiB.\n"
            f"Tips to reduce memory WITHOUT CPU offload:\n"
            f"  - Decrease MAX_SEQ_LEN (e.g., 256 or 192)\n"
            f"  - Increase GRADIENT_ACC_STEPS (e.g., 16)\n"
            f"  - Set ULTRA_LOW_MEM=True (LoRA r=8, q/v only)\n"
            f"  - Ensure no other processes occupy GPU (nvidia-smi)\n"
        )

# Before anything heavy, try to free cached memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Heuristic: require at least ~10 GiB free for 8B + LoRA + checkpointing at seq 384.
# If other jobs occupy memory, this will fail early with clear guidance.
assert_free_mem(min_gib_required=10.0)

# ------------------------
# Tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ------------------------
# Model (GPU only, no offload)
# ------------------------
use_bf16 = torch.cuda.get_device_capability()[0] >= 8  # Ampere+ supports bf16 well
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

# IMPORTANT: keep everything on GPU (no CPU offload)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    device_map={"": "cuda:0"},   # force whole model to GPU
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=False,     # not needed since we don't offload
)

# If we added a new pad token beyond vocab size, resize embeddings (rare)
if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
    model.resize_token_embeddings(len(tokenizer))

# Memory savers on GPU
model.config.use_cache = False
# Prefer SDPA attention (PyTorch built-in, more memory efficient than eager)
try:
    model.config.attn_implementation = "sdpa"
except Exception:
    pass
# Gradient checkpointing (non-reentrant is commonly more memory-friendly)
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    model.gradient_checkpointing_enable()

# ------------------------
# Data
# ------------------------
@dataclass
class MCItem:
    question: str
    options: Dict[str, str]
    answer_letter: str
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path); items=[]; bad=0
    it = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in it:
        q = str(ex.get("question","")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k,v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2: bad+=1; continue
        ans = str(ex.get("answer_idx", ex.get("answer",""))).strip().upper()
        if ans not in opts:
            inv = {v.strip(): k for k,v in opts.items()}
            ans = inv.get(ans,"")
        if ans not in opts: bad+=1; continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad: print(f"[MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_medmcqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path); items=[]; bad=0
    idx_to_letter = {1:"A",2:"B",3:"C",4:"D",5:"E"}; strnum = {"1":"A","2":"B","3":"C","4":"D","5":"E"}
    it = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in it:
        q = str(ex.get("question","")).strip(); opts={}
        if isinstance(ex.get("options"), dict):
            for k,v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A","B","C","D","E"]: opts[kk]=str(v)
        else:
            for L,fld in {"A":"opa","B":"opb","C":"opc","D":"opd","E":"ope"}.items():
                if fld in ex and ex[fld] is not None: opts[L]=str(ex[fld])
        if len(opts) < 2 or not q: bad+=1; continue
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label",""))))
        if isinstance(gold_raw,int): gold = idx_to_letter.get(gold_raw,"")
        else:
            s = str(gold_raw).strip()
            if s in strnum: gold=strnum[s]
            elif len(s)==1 and s.lower() in "abcde": gold=s.upper()
            elif s.upper() in ["A","B","C","D","E"]: gold=s.upper()
            else:
                inv={v.strip():k for k,v in opts.items()}; gold=inv.get(s,"")
        if gold not in opts: bad+=1; continue
        items.append(MCItem(q,opts,gold,str(key)))
    if bad: print(f"[MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path); items=[]; bad=0
    it = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in it:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list): ctx=[str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"}: bad+=1; continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad: print(f"[PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

class AnswerOnlySFTDataset(Dataset):
    """
    Each example is a short chat where the user provides the question and the model must output ONLY the final answer token.
    Loss is applied ONLY on the assistant's final answer tokens (others masked to -100).
    """
    def __init__(self, mc_items: List[MCItem], ynm_items: List[YesNoMaybeItem], split="train", split_ratio=0.8):
        rng = random.Random(RANDOM_SEED)
        mc = mc_items[:]; ynm = ynm_items[:]; rng.shuffle(mc); rng.shuffle(ynm)
        mc_cut = int(len(mc)*split_ratio); ynm_cut = int(len(ynm)*split_ratio)
        if split=="train": self.mc=mc[:mc_cut]; self.ynm=ynm[:ynm_cut]
        else: self.mc=mc[mc_cut:]; self.ynm=ynm[ynm_cut:]
        self.examples=[]
        for it in self.mc:
            letters="".join(sorted(it.options.keys()))
            opts="\n".join([f"{k}. {v}" for k,v in it.options.items()])
            user=("You are answering a multiple-choice medical question.\n"
                  "Return ONLY one uppercase letter.\n\n"
                  f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
                  f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:")
            self.examples.append(("mc", user, it.answer_letter))
        for it in self.ynm:
            ctx="\n".join(f"- {c}" for c in it.contexts[:6])
            user=("You are assessing a biomedical yes/no/maybe question.\n"
                  "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
                  f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:")
            self.examples.append(("ynm", user, it.gold_label))
        rng.shuffle(self.examples)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        _task, user_msg, ans = self.examples[idx]
        msgs=[{"role":"user","content":user_msg},{"role":"assistant","content":ans}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        ans_ids = tokenizer(ans, add_special_tokens=False)["input_ids"]
        N = len(ans_ids)
        keep = N + 1 if len(labels) >= N + 1 else N
        labels[:] = -100
        labels[-keep:] = input_ids[-keep:]
        return {"input_ids": input_ids, "attention_mask": enc["attention_mask"][0], "labels": labels}

# Build datasets
medqa_items    = load_medqa(MEDQA_PATH)
medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)
train_ds = AnswerOnlySFTDataset(medqa_items+medmcqa_items, pubmedqa_items, split="train", split_ratio=0.8)
val_ds   = AnswerOnlySFTDataset(medqa_items+medmcqa_items, pubmedqa_items, split="val",   split_ratio=0.8)

# ------------------------
# Collator: pad input_ids/attention_mask and labels
# ------------------------
class CausalLMAnswerOnlyCollator:
    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.pad_id = tokenizer.pad_token_id
    def __call__(self, features):
        input_ids  = [f["input_ids"] for f in features]
        attn_mask  = [f["attention_mask"] for f in features]
        labels     = [f["labels"] for f in features]
        input_ids  = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        attn_mask  = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        max_len = input_ids.size(1)
        padded=[]
        for y in labels:
            pad_len = max_len - y.size(0)
            if pad_len > 0:
                y = torch.cat([y, torch.full((pad_len,), -100, dtype=y.dtype, device=y.device)])
            padded.append(y)
        labels = torch.stack(padded, dim=0)
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

collator = CausalLMAnswerOnlyCollator(tokenizer)

# ------------------------
# LoRA (GPU only)
# ------------------------
peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, peft_config)

# Some versions require enabling checkpointing again after wrapping
try:
    lora_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except TypeError:
    lora_model.gradient_checkpointing_enable()

# ------------------------
# Training
# ------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    gradient_accumulation_steps=GRADIENT_ACC_STEPS,
    fp16=(not use_bf16),
    bf16=use_bf16,
    logging_steps=LOGGING_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy="epoch",
    save_total_limit=SAVE_TOTAL_LIMIT,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    report_to="none",
    no_cuda=False,  # ensure we do NOT fall back to CPU
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds if EVAL_STRATEGY != "no" else None,
    data_collator=collator,
)

print("Starting LoRA fine-tuning on GPU only...")
trainer.train()
print("Training done.")
trainer.save_model(OUTPUT_DIR)
print(f"LoRA adapters saved to: {OUTPUT_DIR}")

from huggingface_hub import HfApi, create_repo
import os


HF_TOKEN = "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"

REPO_ID = "Easonwangzk/lora-llama31-med-adapter"

OUTPUT_DIR = "lora-med-eval-gpuonly"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is empty. Set it via environment variable or hardcode the string.")

create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

api = HfApi()
api.upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID,
    token=HF_TOKEN,
    repo_type="model",
)

print(f"✅ Uploaded to: https://huggingface.co/{REPO_ID}")

import os, json, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ------------------------
# Config
# ------------------------
MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
USE_CHAT   = True

# ICL shots (set 0 to disable ICL and evaluate zero-shot)
K_SHOTS_MC   = 0
K_SHOTS_YNM  = 0
BALANCE_LABELS = True
RANDOM_SEED  = 42
MAX_SEQ_LEN  = 1024

# Load LoRA adapters from **local** path (your trained OUTPUT_DIR)
ADAPTER_REPO_OR_PATH = "lora-med-eval-gpuonly"  # <- local folder with adapter_config.json & adapter_model.*

# ------------------------
# Dtypes & tokenizer/model
# ------------------------
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

# Tokenizer: ensure we HAVE a pad token
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
tokenizer.padding_side = "left"

pad_added = False
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        pad_added = True

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch_dtype,
    device_map="auto",
)
# If we added a brand-new token, resize embeddings
if pad_added:
    base_model.resize_token_embeddings(len(tokenizer))

# Make sure model knows the pad id (even when using EOS-as-PAD)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.eval()

# ------------------------
# Data containers & loaders
# ------------------------
@dataclass
class MCItem:
    question: str
    options: Dict[str, str]
    answer_letter: str
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k, v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2:
            bad += 1; continue
        ans = ex.get("answer_idx", ex.get("answer", ""))
        ans = str(ans).strip().upper()
        if ans not in opts:
            inv = {v.strip(): k for k, v in opts.items()}
            ans = inv.get(ans, "")
        if ans not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad: print(f"[MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_medmcqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    idx_to_letter = {1:"A",2:"B",3:"C",4:"D",5:"E"}
    strnum_to_letter = {"1":"A","2":"B","3":"C","4":"D","5":"E"}
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question","")).strip()
        opts = {}
        if isinstance(ex.get("options"), dict):
            for k, v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A","B","C","D","E"]:
                    opts[kk] = str(v)
        else:
            for L, fld in {"A":"opa","B":"opb","C":"opc","D":"opd","E":"ope"}.items():
                if fld in ex and ex[fld] is not None:
                    opts[L] = str(ex[fld])
        if len(opts) < 2 or not q:
            bad += 1; continue
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label",""))))
        gold = ""
        if isinstance(gold_raw, int):
            gold = idx_to_letter.get(gold_raw, "")
        else:
            s = str(gold_raw).strip()
            if s in strnum_to_letter:
                gold = strnum_to_letter[s]
            elif len(s)==1 and s.lower() in "abcde":
                gold = s.upper()
            elif s.upper() in ["A","B","C","D","E"]:
                gold = s.upper()
            else:
                inv = {v.strip(): k for k, v in opts.items()}
                gold = inv.get(s, "")
        if gold not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, gold, str(key)))
    if bad: print(f"[MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path)
    items: List[YesNoMaybeItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list):
            ctx = [str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"}:
            bad += 1; continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad: print(f"[PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

# ------------------------
# Prompting (zero-shot and ICL)
# ------------------------
def apply_chat_template(user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg:
        msgs.append({"role":"system","content":system_msg})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt(item: MCItem) -> str:
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k, v in item.options.items()])
    user = (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter.\n\n"
        f"Question:\n{item.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )
    return apply_chat_template(user) if USE_CHAT else user

def pubmedqa_prompt(item: YesNoMaybeItem) -> str:
    ctx = "\n".join(f"- {c}" for c in item.contexts[:6])
    user = (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{item.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )
    return apply_chat_template(user) if USE_CHAT else user

# ICL helpers
rng = torch.Generator().manual_seed(RANDOM_SEED)

def sample_fewshot_mc(pool: List[MCItem], k: int, avoid_key: Optional[str]=None) -> List[MCItem]:
    if not pool or k <= 0: return []
    cand = [it for it in pool if it.source_id != avoid_key]
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label = {}
    for it in cand:
        by_label.setdefault(it.answer_letter, []).append(it)
    for lab in by_label:
        idx = torch.randperm(len(by_label[lab]), generator=rng).tolist()
        by_label[lab] = [by_label[lab][i] for i in idx]
    out, ptr = [], {lab:0 for lab in by_label}
    labs = sorted(by_label.keys())
    while len(out) < min(k, len(cand)) and labs:
        for lab in list(labs):
            if ptr[lab] < len(by_label[lab]) and len(out) < k:
                out.append(by_label[lab][ptr[lab]]); ptr[lab] += 1
            if ptr[lab] >= len(by_label[lab]): labs.remove(lab)
    return out

def sample_fewshot_ynm(pool: List[YesNoMaybeItem], k: int, avoid_key: Optional[str]=None) -> List[YesNoMaybeItem]:
    if not pool or k <= 0: return []
    cand = [it for it in pool if it.source_id != avoid_key]
    by_label = {}
    for it in cand:
        by_label.setdefault(it.gold_label, []).append(it)
    for lab in by_label:
        idx = torch.randperm(len(by_label[lab]), generator=rng).tolist()
        by_label[lab] = [by_label[lab][i] for i in idx]
    out, ptr = [], {lab:0 for lab in by_label}
    labs = sorted(by_label.keys())
    while len(out) < min(k, len(cand)) and labs:
        for lab in list(labs):
            if ptr[lab] < len(by_label[lab]) and len(out) < k:
                out.append(by_label[lab][ptr[lab]]); ptr[lab] += 1
            if ptr[lab] >= len(by_label[lab]): labs.remove(lab)
    return out

def apply_chat_template_icl(demos: List[Tuple[str, str]], user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg:
        msgs.append({"role":"system","content":system_msg})
    for du, da in demos:
        msgs.append({"role":"user","content":du})
        msgs.append({"role":"assistant","content":da})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt_icl(item: MCItem, pool: List[MCItem], k: int) -> str:
    demos_items = sample_fewshot_mc(pool, k, avoid_key=item.source_id)
    demos = []
    for d in demos_items:
        letters = "".join(sorted(d.options.keys()))
        opts = "\n".join([f"{k}. {v}" for k, v in d.options.items()])
        du = (
            "You are answering a multiple-choice medical question.\n"
            "Return ONLY one uppercase letter.\n\n"
            f"Question:\n{d.question}\n\nOptions:\n{opts}\n\n"
            f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
        )
        demos.append((du, d.answer_letter))
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k, v in item.options.items()])
    user = (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter.\n\n"
        f"Question:\n{item.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )
    return apply_chat_template_icl(demos, user)

def pubmedqa_prompt_icl(item: YesNoMaybeItem, pool: List[YesNoMaybeItem], k: int) -> str:
    demos_items = sample_fewshot_ynm(pool, k, avoid_key=item.source_id)
    demos = []
    for d in demos_items:
        ctx = "\n".join(f"- {c}" for c in d.contexts[:6])
        du = (
            "You are assessing a biomedical yes/no/maybe question.\n"
            "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
            f"Question:\n{d.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
        )
        demos.append((du, d.gold_label))
    ctx = "\n".join(f"- {c}" for c in item.contexts[:6])
    user = (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{item.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )
    return apply_chat_template_icl(demos, user)

# ------------------------
# Generation, parsing, evaluation
# ------------------------
@torch.no_grad()
def generate_answer(model, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return gen.split("Answer:")[-1].strip()

LETTER_RE = re.compile(r"\b([A-E])\b")
YNM_RE    = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)

def parse_mc_letter(text: str, allowed: List[str]) -> Optional[str]:
    m = LETTER_RE.search(text.upper())
    if not m: return None
    cand = m.group(1)
    return cand if cand in allowed else None

def parse_ynm(text: str) -> Optional[str]:
    m = YNM_RE.search(text)
    return m.group(1).lower() if m else None

def eval_mcq(model, items: List[MCItem], desc: str,
             icl_pool: Optional[List[MCItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        prompt = mc_prompt_icl(it, icl_pool, k=k_shots) if (k_shots>0 and icl_pool) else mc_prompt(it)
        out = generate_answer(model, prompt)
        allowed = sorted(list(it.options.keys()))
        pred = parse_mc_letter(out, allowed) or ""
        used += 1; correct += int(pred == it.answer_letter)
    return correct / max(1, used)

def eval_pubmedqa(model, items: List[YesNoMaybeItem], desc: str,
                  icl_pool: Optional[List[YesNoMaybeItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        prompt = pubmedqa_prompt_icl(it, icl_pool, k=k_shots) if (k_shots>0 and icl_pool) else pubmedqa_prompt(it)
        out = generate_answer(model, prompt)
        pred = parse_ynm(out) or ""
        used += 1; correct += int(pred == it.gold_label)
    return correct / max(1, used)

def run_full_eval(tag: str, model) -> Tuple[float, float, float, float]:
    medqa_items    = load_medqa(MEDQA_PATH)
    medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
    pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)

    medqa_pool    = medqa_items
    medmcqa_pool  = medmcqa_items
    pubmedqa_pool = pubmedqa_items

    medqa_acc    = eval_mcq(model, medqa_items,   f"[{tag}] MedQA (k={K_SHOTS_MC})",     icl_pool=medqa_pool,   k_shots=K_SHOTS_MC)
    medmcqa_acc  = eval_mcq(model, medmcqa_items, f"[{tag}] MedMCQA (k={K_SHOTS_MC})",   icl_pool=medmcqa_pool, k_shots=K_SHOTS_MC)
    pubmedqa_acc = eval_pubmedqa(model, pubmedqa_items, f"[{tag}] PubMedQA (k={K_SHOTS_YNM})", icl_pool=pubmedqa_pool, k_shots=K_SHOTS_YNM)
    macro = (medqa_acc + medmcqa_acc + pubmedqa_acc) / 3.0

    print(f"\n[{tag}] MedQA acc:    {medqa_acc:.3f}")
    print(f"[{tag}] MedMCQA acc:  {medmcqa_acc:.3f}")
    print(f"[{tag}] PubMedQA acc: {pubmedqa_acc:.3f}")
    print("-"*60)
    print(f"[{tag}] Macro-average: {macro:.3f}\n")
    return medqa_acc, medmcqa_acc, pubmedqa_acc, macro

# ------------------------
# Evaluate base and local LoRA
# ------------------------
print("\n=== EVALUATING BASE MODEL ===")
_ = run_full_eval("BASE", base_model)

if ADAPTER_REPO_OR_PATH:
    # Quick local check
    cfg_path = os.path.join(ADAPTER_REPO_OR_PATH, "adapter_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in '{ADAPTER_REPO_OR_PATH}'. "
            f"Ensure this is your trained OUTPUT_DIR with LoRA files."
        )
    print("\n=== LOADING LOCAL LoRA ADAPTERS ===")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO_OR_PATH)
    lora_model.eval()
    print("=== EVALUATING LoRA MODEL ===")
    _ = run_full_eval("LoRA", lora_model)
else:
    print("No adapters specified; skipped LoRA evaluation.")
