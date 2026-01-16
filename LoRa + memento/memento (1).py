from huggingface_hub import login
login("hf_....")

from huggingface_hub import HfApi, HfFolder, upload_file


api = HfApi()

repo_id = "Easonwangzk/MedLLM_Data"

upload_file(
    path_or_fileobj="medqa_50.json",
    path_in_repo="datasets/medqa_50.json",
    repo_id=repo_id,
    repo_type="dataset",
)

upload_file(
    path_or_fileobj="medmcqa_50.json",
    path_in_repo="datasets/medmcqa_50.json",
    repo_id=repo_id,
    repo_type="dataset",
)

upload_file(
    path_or_fileobj="pubmedqa_50.json",
    path_in_repo="datasets/pubmedqa_50.json",
    repo_id=repo_id,
    repo_type="dataset",
)

import json, re, os, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# Config

MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

# Base model and optional LoRA adapter
MODEL_REPO    = "meta-llama/Llama-3.1-8B-Instruct"
TEST_LORA_ADAPTER = True
ADAPTER_REPO      = "Easonwangzk/lora-llama31-med-adapter"

# Prompting
USE_CHAT   = True
DET_SAMPLE = False 

# ICL
K_SHOTS_MC  = 2
K_SHOTS_YNM = 2
RANDOM_SEED = 42
BALANCE_LABELS = True

# MemoryBank (Memento-style)
MEM_CAPACITY             = 2000
USE_ONLY_SUCCESS         = True      # store only correct cases
ENABLE_ONLINE_WRITEBACK  = True      # set False to disable writeback during eval
CASE_BANK_PATH           = "case_bank.json"  # persisted case bank file

# Warm seed (use dev/train items ideally; here we use first N of loaded sets for demo)
WARM_SEED_MC_MAX  = 8
WARM_SEED_YNM_MAX = 8


# dtype
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, torch_dtype=torch_dtype, device_map="auto"
)
base_model.eval()
print(f"[INFO] Model loaded: {MODEL_REPO}")


# Retrieval backends for MemoryBank

# Prefer Sentence-BERT and fallback to TF-IDF if unavailable.
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _USE_EMB = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _VEC = TfidfVectorizer(max_features=4096)
    _USE_EMB = False


# Data containers

@dataclass
class MCItem:
    question: str
    options: Dict[str, str]       # keys "A".."E"
    answer_letter: str            # gold letter
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str               # "yes"/"no"/"maybe"
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========================
# Loaders
# ========================
def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k, v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2 or not q:
            bad += 1; continue
        ans = str(ex.get("answer_idx", ex.get("answer", ""))).strip().upper()
        if ans not in opts:
            inv = {v.strip(): k for k, v in opts.items()}
            ans = inv.get(ans, "")
        if ans not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad: print(f"[WARN] [MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
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
        opts: Dict[str,str] = {}
        if isinstance(ex.get("options"), dict):
            for k,v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A","B","C","D","E"]: opts[kk] = str(v)
        else:
            for L, fld in {"A":"opa","B":"opb","C":"opc","D":"opd","E":"ope"}.items():
                if fld in ex and ex[fld] is not None: opts[L] = str(ex[fld])
        if len(opts) < 2 or not q:
            bad += 1; continue
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label",""))))
        gold = ""
        if isinstance(gold_raw, int):
            gold = idx_to_letter.get(gold_raw, "")
        else:
            s = str(gold_raw).strip()
            if s in strnum_to_letter: gold = strnum_to_letter[s]
            elif len(s)==1 and s.lower() in "abcde": gold = s.upper()
            elif s.upper() in ["A","B","C","D","E"]: gold = s.upper()
            else:
                inv = {v.strip(): k for k, v in opts.items()}
                gold = inv.get(s, "")
        if gold not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, gold, str(key)))
    if bad: print(f"[WARN] [MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path)
    items: List[YesNoMaybeItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list): ctx = [str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"} or not q:
            bad += 1; continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad: print(f"[WARN] [PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

# ========================
# Prompt builders
# ========================
def apply_chat_template(user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def apply_chat_template_icl(demos: List[Tuple[str,str]], user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    for du, da in demos:
        msgs.append({"role":"user","content":du})
        msgs.append({"role":"assistant","content":da})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt(item: MCItem) -> str:
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in item.options.items()])
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

def mc_demo_user(it: MCItem) -> str:
    letters = "".join(sorted(it.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
    return (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter.\n\n"
        f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )

def mc_demo_assistant(it: MCItem) -> str:
    return it.answer_letter

def pubmedqa_demo_user(it: YesNoMaybeItem) -> str:
    ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
    return (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )

def pubmedqa_demo_assistant(it: YesNoMaybeItem) -> str:
    return it.gold_label

# ========================
# Memory Bank (Memento-style)
# ========================
class MemoryBank:
    """Non-parametric store for solved cases + similarity retrieval."""
    def __init__(self, use_only_success: bool = True, capacity: int = 2000):
        self.use_only_success = use_only_success
        self.capacity = capacity
        self.cases: List[Dict[str, Any]] = []
        self._fit_ready = False

    def add(self, task_type: str, query_text: str, demo_user: str,
            demo_assistant: str, success: bool):
        if self.use_only_success and not success:
            return
        self.cases.append({
            "task_type": task_type,
            "query_text": (query_text or "").strip(),
            "demo_user": demo_user,
            "demo_assistant": demo_assistant,
        })
        if len(self.cases) > self.capacity:
            self.cases = self.cases[-self.capacity:]
        self._fit_ready = False

    def _ensure_index(self):
        texts = [c["query_text"] for c in self.cases]
        if _USE_EMB:
            self._emb = _SBERT.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 384))
        else:
            self._mat = _VEC.fit_transform(texts) if texts else None
        self._fit_ready = True

    def _similarities(self, query_text: str) -> np.ndarray:
        if not self.cases:
            return np.zeros((0,))
        if not self._fit_ready:
            self._ensure_index()
        if _USE_EMB:
            q = _SBERT.encode([query_text], normalize_embeddings=True)
            return (self._emb @ q[0]).astype(np.float32)
        else:
            q = _VEC.transform([query_text])
            return (self._mat @ q.T).toarray().ravel().astype(np.float32)

    def retrieve(self, task_type: str, query_text: str, top_k: int) -> List[Tuple[str, str]]:
        if not self.cases or top_k <= 0:
            return []
        sims = self._similarities(query_text)
        idx = np.argsort(-sims)
        demos: List[Tuple[str,str]] = []
        for i in idx:
            c = self.cases[i]
            if c["task_type"] != task_type:
                continue
            demos.append((c["demo_user"], c["demo_assistant"]))
            if len(demos) >= top_k:
                break
        return demos

CASE_BANK = MemoryBank(use_only_success=USE_ONLY_SUCCESS, capacity=MEM_CAPACITY)

# ========================
# Case Bank persistence + warm seed
# ========================
def save_case_bank(path: str = CASE_BANK_PATH):
    """Persist the in-memory case bank to JSON."""
    data = [{
        "task_type": c["task_type"],
        "query_text": c["query_text"],
        "demo_user": c["demo_user"],
        "demo_assistant": c["demo_assistant"],
    } for c in CASE_BANK.cases]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_case_bank(path: str = CASE_BANK_PATH):
    """Load a previously saved case bank JSON (if present)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    CASE_BANK.cases = list(data)
    CASE_BANK._fit_ready = False  # rebuild index lazily

def warm_seed_case_bank_mc(mc_items: List[MCItem], max_n: int = 8):
    """Preload a few strong MC demos (ideally from train/dev)."""
    for it in mc_items[:max_n]:
        CASE_BANK.add(
            task_type="mc",
            query_text=it.question,
            demo_user=mc_demo_user(it),
            demo_assistant=it.answer_letter,
            success=True,
        )

def warm_seed_case_bank_ynm(ynm_items: List[YesNoMaybeItem], max_n: int = 8):
    """Preload a few strong YNM demos (ideally from train/dev)."""
    for it in ynm_items[:max_n]:
        CASE_BANK.add(
            task_type="ynm",
            query_text=it.question,
            demo_user=pubmedqa_demo_user(it),
            demo_assistant=it.gold_label,
            success=True,
        )

# ========================
# ICL fallback sampling (when bank is short)
# ========================
rng = torch.Generator().manual_seed(RANDOM_SEED)

def sample_fewshot_mc(pool: List[MCItem], k: int, avoid_key: Optional[str]=None) -> List[MCItem]:
    if not pool or k <= 0: return []
    cand = [it for it in pool if it.source_id != avoid_key]
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[MCItem]] = {}
    for it in cand: by_label.setdefault(it.answer_letter, []).append(it)
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
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[YesNoMaybeItem]] = {}
    for it in cand: by_label.setdefault(it.gold_label, []).append(it)
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

# ========================
# Generation & parsing
# ========================
@torch.no_grad()
def generate_answer_with(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=DET_SAMPLE, pad_token_id=tokenizer.eos_token_id
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

def _retry_if_malformed(model, tokenizer, prompt: str,
                        allowed_letters: Optional[List[str]] = None,
                        ynm: bool = False):
    """Deterministic single retry on malformed output."""
    out = generate_answer_with(model, tokenizer, prompt)
    parsed = parse_ynm(out) if ynm else parse_mc_letter(out, allowed_letters or [])
    if parsed: return out, parsed
    stricter = prompt + "\nStrictly output only the final token. No explanations. Answer:"
    out2 = generate_answer_with(model, tokenizer, stricter)
    parsed2 = parse_ynm(out2) if ynm else parse_mc_letter(out2, allowed_letters or [])
    return (out2, parsed2) if parsed2 else (out, None)

# ========================
# Evaluation (MemoryBank retrieval + fallback + online writeback)
# ========================
def eval_mcq_with(model, tokenizer, items: List[MCItem], desc: str,
                  icl_pool: Optional[List[MCItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("mc", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_mc(icl_pool, need, avoid_key=it.source_id)
            demos += [(mc_demo_user(d), mc_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            letters = "".join(sorted(it.options.keys()))
            opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
            user_msg = (
                "You are answering a multiple-choice medical question.\n"
                "Return ONLY one uppercase letter.\n\n"
                f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
                f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = mc_prompt(it)
        allowed = sorted(list(it.options.keys()))
        out, pred = _retry_if_malformed(model, tokenizer, prompt, allowed_letters=allowed, ynm=False)
        used += 1
        is_correct = (pred or "") == it.answer_letter
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("mc", it.question, mc_demo_user(it), it.answer_letter, success=True)
    return correct / max(1, used)

def eval_pubmedqa_with(model, tokenizer, items: List[YesNoMaybeItem], desc: str,
                       icl_pool: Optional[List[YesNoMaybeItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("ynm", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_ynm(icl_pool, need, avoid_key=it.source_id)
            demos += [(pubmedqa_demo_user(d), pubmedqa_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
            user_msg = (
                "You are assessing a biomedical yes/no/maybe question.\n"
                "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
                f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = pubmedqa_prompt(it)
        out, pred = _retry_if_malformed(model, tokenizer, prompt, ynm=True)
        used += 1
        is_correct = (pred or "") == it.gold_label
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("ynm", it.question, pubmedqa_demo_user(it), it.gold_label, success=True)
    return correct / max(1, used)

# ========================
# Run helpers
# ========================
def run_all(tag: str, model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            k_mc: int, k_ynm: int):
    medqa_acc    = eval_mcq_with(model, tokenizer, medqa_items,   f"[{tag}] MedQA (k={k_mc})",
                                 icl_pool=medqa_pool,   k_shots=k_mc)
    medmcqa_acc  = eval_mcq_with(model, tokenizer, medmcqa_items, f"[{tag}] MedMCQA (k={k_mc})",
                                 icl_pool=medmcqa_pool, k_shots=k_mc)
    pubmedqa_acc = eval_pubmedqa_with(model, tokenizer, pubmedqa_items, f"[{tag}] PubMedQA (k={k_ynm})",
                                      icl_pool=pubmedqa_pool, k_shots=k_ynm)
    macro_acc = (medqa_acc + medmcqa_acc + pubmedqa_acc) / 3.0
    print(f"\n[{tag}] MedQA acc:    {medqa_acc:.3f}")
    print(f"[{tag}] MedMCQA acc:  {medmcqa_acc:.3f}")
    print(f"[{tag}] PubMedQA acc: {pubmedqa_acc:.3f}")
    print("-" * 52)
    print(f"[{tag}] Macro avg:    {macro_acc:.3f}")
    return medqa_acc, medmcqa_acc, pubmedqa_acc, macro_acc

# ========================
# Main
# ========================
def main():
    # Load datasets
    medqa_items    = load_medqa(MEDQA_PATH)
    medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
    pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)

    # Load existing case bank (if any), then warm seed a few demos
    load_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK loaded: {len(CASE_BANK.cases)} cases")

    # Warm seed (ideally use train/dev items; here we use first N for demo)
    warm_seed_case_bank_mc(medqa_items, max_n=WARM_SEED_MC_MAX)
    warm_seed_case_bank_ynm(pubmedqa_items, max_n=WARM_SEED_YNM_MAX)
    print(f"[INFO] CASE_BANK after warm seed: {len(CASE_BANK.cases)} cases")

    # Few-shot pools (fallback)
    medqa_pool, medmcqa_pool, pubmedqa_pool = medqa_items, medmcqa_items, pubmedqa_items

    # Baseline round
    run_all("BASE", base_model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            K_SHOTS_MC, K_SHOTS_YNM)

    # Persist bank after baseline
    save_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK saved (after BASE): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

    # Optional LoRA round
    if TEST_LORA_ADAPTER:
        print(f"[INFO] Loading LoRA adapter: {ADAPTER_REPO}")
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, torch_dtype=torch_dtype)
        lora_model.eval()
        run_all("LoRA", lora_model, tokenizer,
                medqa_items, medmcqa_items, pubmedqa_items,
                medqa_pool, medmcqa_pool, pubmedqa_pool,
                K_SHOTS_MC, K_SHOTS_YNM)
        save_case_bank(CASE_BANK_PATH)
        print(f"[INFO] CASE_BANK saved (after LoRA): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

if __name__ == "__main__":
    main()

import json, re, os, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# ========================
# Config
# ========================
# Datasets
MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

# Base model and optional LoRA adapter
MODEL_REPO    = "meta-llama/Llama-3.1-8B-Instruct"
TEST_LORA_ADAPTER = True
ADAPTER_REPO      = "Easonwangzk/lora-llama31-med-adapter"

# Prompting
USE_CHAT   = True     # use model's chat template
DET_SAMPLE = False    # greedy decoding for determinism

# ICL
K_SHOTS_MC  = 3
K_SHOTS_YNM = 3
RANDOM_SEED = 42
BALANCE_LABELS = True

# MemoryBank (Memento-style)
MEM_CAPACITY             = 2000
USE_ONLY_SUCCESS         = True      # store only correct cases
ENABLE_ONLINE_WRITEBACK  = True      # set False to disable writeback during eval
CASE_BANK_PATH           = "case_bank.json"  # persisted case bank file

# Warm seed (use dev/train items ideally; here we use first N of loaded sets for demo)
WARM_SEED_MC_MAX  = 8
WARM_SEED_YNM_MAX = 8

# ========================
# Device / dtype
# ========================
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, torch_dtype=torch_dtype, device_map="auto"
)
base_model.eval()
print(f"[INFO] Model loaded: {MODEL_REPO}")

# ========================
# Optional retrieval backends for MemoryBank
# ========================
# Prefer Sentence-BERT; fallback to TF-IDF if unavailable.
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _USE_EMB = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _VEC = TfidfVectorizer(max_features=4096)
    _USE_EMB = False

# ========================
# Data containers
# ========================
@dataclass
class MCItem:
    question: str
    options: Dict[str, str]       # keys "A".."E"
    answer_letter: str            # gold letter
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str               # "yes"/"no"/"maybe"
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========================
# Loaders
# ========================
def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k, v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2 or not q:
            bad += 1; continue
        ans = str(ex.get("answer_idx", ex.get("answer", ""))).strip().upper()
        if ans not in opts:
            inv = {v.strip(): k for k, v in opts.items()}
            ans = inv.get(ans, "")
        if ans not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad: print(f"[WARN] [MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
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
        opts: Dict[str,str] = {}
        if isinstance(ex.get("options"), dict):
            for k,v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A","B","C","D","E"]: opts[kk] = str(v)
        else:
            for L, fld in {"A":"opa","B":"opb","C":"opc","D":"opd","E":"ope"}.items():
                if fld in ex and ex[fld] is not None: opts[L] = str(ex[fld])
        if len(opts) < 2 or not q:
            bad += 1; continue
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label",""))))
        gold = ""
        if isinstance(gold_raw, int):
            gold = idx_to_letter.get(gold_raw, "")
        else:
            s = str(gold_raw).strip()
            if s in strnum_to_letter: gold = strnum_to_letter[s]
            elif len(s)==1 and s.lower() in "abcde": gold = s.upper()
            elif s.upper() in ["A","B","C","D","E"]: gold = s.upper()
            else:
                inv = {v.strip(): k for k, v in opts.items()}
                gold = inv.get(s, "")
        if gold not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, gold, str(key)))
    if bad: print(f"[WARN] [MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path)
    items: List[YesNoMaybeItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list): ctx = [str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"} or not q:
            bad += 1; continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad: print(f"[WARN] [PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

# ========================
# Prompt builders
# ========================
def apply_chat_template(user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def apply_chat_template_icl(demos: List[Tuple[str,str]], user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    for du, da in demos:
        msgs.append({"role":"user","content":du})
        msgs.append({"role":"assistant","content":da})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt(item: MCItem) -> str:
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in item.options.items()])
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

def mc_demo_user(it: MCItem) -> str:
    letters = "".join(sorted(it.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
    return (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter.\n\n"
        f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )

def mc_demo_assistant(it: MCItem) -> str:
    return it.answer_letter

def pubmedqa_demo_user(it: YesNoMaybeItem) -> str:
    ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
    return (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )

def pubmedqa_demo_assistant(it: YesNoMaybeItem) -> str:
    return it.gold_label

# ========================
# Memory Bank (Memento-style)
# ========================
class MemoryBank:
    """Non-parametric store for solved cases + similarity retrieval."""
    def __init__(self, use_only_success: bool = True, capacity: int = 2000):
        self.use_only_success = use_only_success
        self.capacity = capacity
        self.cases: List[Dict[str, Any]] = []
        self._fit_ready = False

    def add(self, task_type: str, query_text: str, demo_user: str,
            demo_assistant: str, success: bool):
        if self.use_only_success and not success:
            return
        self.cases.append({
            "task_type": task_type,
            "query_text": (query_text or "").strip(),
            "demo_user": demo_user,
            "demo_assistant": demo_assistant,
        })
        if len(self.cases) > self.capacity:
            self.cases = self.cases[-self.capacity:]
        self._fit_ready = False

    def _ensure_index(self):
        texts = [c["query_text"] for c in self.cases]
        if _USE_EMB:
            self._emb = _SBERT.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 384))
        else:
            self._mat = _VEC.fit_transform(texts) if texts else None
        self._fit_ready = True

    def _similarities(self, query_text: str) -> np.ndarray:
        if not self.cases:
            return np.zeros((0,))
        if not self._fit_ready:
            self._ensure_index()
        if _USE_EMB:
            q = _SBERT.encode([query_text], normalize_embeddings=True)
            return (self._emb @ q[0]).astype(np.float32)
        else:
            q = _VEC.transform([query_text])
            return (self._mat @ q.T).toarray().ravel().astype(np.float32)

    def retrieve(self, task_type: str, query_text: str, top_k: int) -> List[Tuple[str, str]]:
        if not self.cases or top_k <= 0:
            return []
        sims = self._similarities(query_text)
        idx = np.argsort(-sims)
        demos: List[Tuple[str,str]] = []
        for i in idx:
            c = self.cases[i]
            if c["task_type"] != task_type:
                continue
            demos.append((c["demo_user"], c["demo_assistant"]))
            if len(demos) >= top_k:
                break
        return demos

CASE_BANK = MemoryBank(use_only_success=USE_ONLY_SUCCESS, capacity=MEM_CAPACITY)

# ========================
# Case Bank persistence + warm seed
# ========================
def save_case_bank(path: str = CASE_BANK_PATH):
    """Persist the in-memory case bank to JSON."""
    data = [{
        "task_type": c["task_type"],
        "query_text": c["query_text"],
        "demo_user": c["demo_user"],
        "demo_assistant": c["demo_assistant"],
    } for c in CASE_BANK.cases]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_case_bank(path: str = CASE_BANK_PATH):
    """Load a previously saved case bank JSON (if present)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    CASE_BANK.cases = list(data)
    CASE_BANK._fit_ready = False  # rebuild index lazily

def warm_seed_case_bank_mc(mc_items: List[MCItem], max_n: int = 8):
    """Preload a few strong MC demos (ideally from train/dev)."""
    for it in mc_items[:max_n]:
        CASE_BANK.add(
            task_type="mc",
            query_text=it.question,
            demo_user=mc_demo_user(it),
            demo_assistant=it.answer_letter,
            success=True,
        )

def warm_seed_case_bank_ynm(ynm_items: List[YesNoMaybeItem], max_n: int = 8):
    """Preload a few strong YNM demos (ideally from train/dev)."""
    for it in ynm_items[:max_n]:
        CASE_BANK.add(
            task_type="ynm",
            query_text=it.question,
            demo_user=pubmedqa_demo_user(it),
            demo_assistant=it.gold_label,
            success=True,
        )

# ========================
# ICL fallback sampling (when bank is short)
# ========================
rng = torch.Generator().manual_seed(RANDOM_SEED)

def sample_fewshot_mc(pool: List[MCItem], k: int, avoid_key: Optional[str]=None) -> List[MCItem]:
    if not pool or k <= 0: return []
    cand = [it for it in pool if it.source_id != avoid_key]
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[MCItem]] = {}
    for it in cand: by_label.setdefault(it.answer_letter, []).append(it)
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
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[YesNoMaybeItem]] = {}
    for it in cand: by_label.setdefault(it.gold_label, []).append(it)
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

# ========================
# Generation & parsing
# ========================
@torch.no_grad()
def generate_answer_with(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=DET_SAMPLE, pad_token_id=tokenizer.eos_token_id
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

def _retry_if_malformed(model, tokenizer, prompt: str,
                        allowed_letters: Optional[List[str]] = None,
                        ynm: bool = False):
    """Deterministic single retry on malformed output."""
    out = generate_answer_with(model, tokenizer, prompt)
    parsed = parse_ynm(out) if ynm else parse_mc_letter(out, allowed_letters or [])
    if parsed: return out, parsed
    stricter = prompt + "\nStrictly output only the final token. No explanations. Answer:"
    out2 = generate_answer_with(model, tokenizer, stricter)
    parsed2 = parse_ynm(out2) if ynm else parse_mc_letter(out2, allowed_letters or [])
    return (out2, parsed2) if parsed2 else (out, None)

# ========================
# Evaluation (MemoryBank retrieval + fallback + online writeback)
# ========================
def eval_mcq_with(model, tokenizer, items: List[MCItem], desc: str,
                  icl_pool: Optional[List[MCItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("mc", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_mc(icl_pool, need, avoid_key=it.source_id)
            demos += [(mc_demo_user(d), mc_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            letters = "".join(sorted(it.options.keys()))
            opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
            user_msg = (
                "You are answering a multiple-choice medical question.\n"
                "Return ONLY one uppercase letter.\n\n"
                f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
                f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = mc_prompt(it)
        allowed = sorted(list(it.options.keys()))
        out, pred = _retry_if_malformed(model, tokenizer, prompt, allowed_letters=allowed, ynm=False)
        used += 1
        is_correct = (pred or "") == it.answer_letter
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("mc", it.question, mc_demo_user(it), it.answer_letter, success=True)
    return correct / max(1, used)

def eval_pubmedqa_with(model, tokenizer, items: List[YesNoMaybeItem], desc: str,
                       icl_pool: Optional[List[YesNoMaybeItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("ynm", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_ynm(icl_pool, need, avoid_key=it.source_id)
            demos += [(pubmedqa_demo_user(d), pubmedqa_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
            user_msg = (
                "You are assessing a biomedical yes/no/maybe question.\n"
                "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
                f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = pubmedqa_prompt(it)
        out, pred = _retry_if_malformed(model, tokenizer, prompt, ynm=True)
        used += 1
        is_correct = (pred or "") == it.gold_label
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("ynm", it.question, pubmedqa_demo_user(it), it.gold_label, success=True)
    return correct / max(1, used)

# ========================
# Run helpers
# ========================
def run_all(tag: str, model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            k_mc: int, k_ynm: int):
    medqa_acc    = eval_mcq_with(model, tokenizer, medqa_items,   f"[{tag}] MedQA (k={k_mc})",
                                 icl_pool=medqa_pool,   k_shots=k_mc)
    medmcqa_acc  = eval_mcq_with(model, tokenizer, medmcqa_items, f"[{tag}] MedMCQA (k={k_mc})",
                                 icl_pool=medmcqa_pool, k_shots=k_mc)
    pubmedqa_acc = eval_pubmedqa_with(model, tokenizer, pubmedqa_items, f"[{tag}] PubMedQA (k={k_ynm})",
                                      icl_pool=pubmedqa_pool, k_shots=k_ynm)
    macro_acc = (medqa_acc + medmcqa_acc + pubmedqa_acc) / 3.0
    print(f"\n[{tag}] MedQA acc:    {medqa_acc:.3f}")
    print(f"[{tag}] MedMCQA acc:  {medmcqa_acc:.3f}")
    print(f"[{tag}] PubMedQA acc: {pubmedqa_acc:.3f}")
    print("-" * 52)
    print(f"[{tag}] Macro avg:    {macro_acc:.3f}")
    return medqa_acc, medmcqa_acc, pubmedqa_acc, macro_acc

# ========================
# Main
# ========================
def main():
    # Load datasets
    medqa_items    = load_medqa(MEDQA_PATH)
    medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
    pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)

    # Load existing case bank (if any), then warm seed a few demos
    load_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK loaded: {len(CASE_BANK.cases)} cases")

    # Warm seed (ideally use train/dev items; here we use first N for demo)
    warm_seed_case_bank_mc(medqa_items, max_n=WARM_SEED_MC_MAX)
    warm_seed_case_bank_ynm(pubmedqa_items, max_n=WARM_SEED_YNM_MAX)
    print(f"[INFO] CASE_BANK after warm seed: {len(CASE_BANK.cases)} cases")

    # Few-shot pools (fallback)
    medqa_pool, medmcqa_pool, pubmedqa_pool = medqa_items, medmcqa_items, pubmedqa_items

    # Baseline round
    run_all("BASE", base_model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            K_SHOTS_MC, K_SHOTS_YNM)

    # Persist bank after baseline
    save_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK saved (after BASE): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

    # Optional LoRA round
    if TEST_LORA_ADAPTER:
        print(f"[INFO] Loading LoRA adapter: {ADAPTER_REPO}")
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, torch_dtype=torch_dtype)
        lora_model.eval()
        run_all("LoRA", lora_model, tokenizer,
                medqa_items, medmcqa_items, pubmedqa_items,
                medqa_pool, medmcqa_pool, pubmedqa_pool,
                K_SHOTS_MC, K_SHOTS_YNM)
        save_case_bank(CASE_BANK_PATH)
        print(f"[INFO] CASE_BANK saved (after LoRA): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

if __name__ == "__main__":
    main()

"""# Output CSV"""

import json, re, os, math, csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np

# ========================
# Config
# ========================
# Datasets
MEDQA_PATH    = "medqa_50.json"
MEDMCQA_PATH  = "medmcqa_50.json"
PUBMEDQA_PATH = "pubmedqa_50.json"

# Base model and optional LoRA adapter
MODEL_REPO    = "meta-llama/Llama-3.1-8B-Instruct"
TEST_LORA_ADAPTER = True
ADAPTER_REPO      = "Easonwangzk/lora-llama31-med-adapter"

# Prompting
USE_CHAT   = True     # use model's chat template
DET_SAMPLE = False    # greedy decoding for determinism

# ICL
K_SHOTS_MC  = 2
K_SHOTS_YNM = 2
RANDOM_SEED = 42
BALANCE_LABELS = True

# MemoryBank (Memento-style)
MEM_CAPACITY             = 2000
USE_ONLY_SUCCESS         = True      # store only correct cases
ENABLE_ONLINE_WRITEBACK  = True      # set False to disable writeback during eval
CASE_BANK_PATH           = "case_bank.json"  # persisted case bank file

# Warm seed (use dev/train items ideally; here we use first N of loaded sets for demo)
WARM_SEED_MC_MAX  = 8
WARM_SEED_YNM_MAX = 8

# CSV output
CSV_PATH = "eval_results.csv"

# ========================
# Device / dtype
# ========================
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_REPO, torch_dtype=torch_dtype, device_map="auto"
)
base_model.eval()
print(f"[INFO] Model loaded: {MODEL_REPO}")

# ========================
# Optional retrieval backends for MemoryBank
# ========================
# Prefer Sentence-BERT; fallback to TF-IDF if unavailable.
try:
    from sentence_transformers import SentenceTransformer
    _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _USE_EMB = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _VEC = TfidfVectorizer(max_features=4096)
    _USE_EMB = False

# ========================
# Data containers
# ========================
@dataclass
class MCItem:
    question: str
    options: Dict[str, str]       # keys "A".."E"
    answer_letter: str            # gold letter
    source_id: Optional[str] = None

@dataclass
class YesNoMaybeItem:
    question: str
    contexts: List[str]
    gold_label: str               # "yes"/"no"/"maybe"
    source_id: Optional[str] = None

def _read_json_any(path: str) -> Union[dict, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========================
# Loaders
# ========================
def load_medqa(path: str) -> List[MCItem]:
    raw = _read_json_any(path)
    items: List[MCItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("question", "")).strip()
        opts_in = ex.get("options", {})
        opts = {k.upper(): str(v) for k, v in opts_in.items() if k.upper() in ["A","B","C","D","E"]}
        if len(opts) < 2 or not q:
            bad += 1; continue
        ans = str(ex.get("answer_idx", ex.get("answer", ""))).strip().upper()
        if ans not in opts:
            inv = {v.strip(): k for k, v in opts.items()}
            ans = inv.get(ans, "")
        if ans not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, ans, str(key)))
    if bad: print(f"[WARN] [MedQA] skipped {bad} malformed item(s). Using {len(items)}.")
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
        opts: Dict[str,str] = {}
        if isinstance(ex.get("options"), dict):
            for k,v in ex["options"].items():
                kk = str(k).strip().upper()
                if kk in ["A","B","C","D","E"]: opts[kk] = str(v)
        else:
            for L, fld in {"A":"opa","B":"opb","C":"opc","D":"opd","E":"ope"}.items():
                if fld in ex and ex[fld] is not None: opts[L] = str(ex[fld])
        if len(opts) < 2 or not q:
            bad += 1; continue
        gold_raw = ex.get("cop", ex.get("answer_idx", ex.get("answer", ex.get("label",""))))
        gold = ""
        if isinstance(gold_raw, int):
            gold = idx_to_letter.get(gold_raw, "")
        else:
            s = str(gold_raw).strip()
            if s in strnum_to_letter: gold = strnum_to_letter[s]
            elif len(s)==1 and s.lower() in "abcde": gold = s.upper()
            elif s.upper() in ["A","B","C","D","E"]: gold = s.upper()
            else:
                inv = {v.strip(): k for k, v in opts.items()}
                gold = inv.get(s, "")
        if gold not in opts:
            bad += 1; continue
        items.append(MCItem(q, opts, gold, str(key)))
    if bad: print(f"[WARN] [MedMCQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

def load_pubmedqa(path: str) -> List[YesNoMaybeItem]:
    raw = _read_json_any(path)
    items: List[YesNoMaybeItem] = []
    bad = 0
    iterator = raw.items() if isinstance(raw, dict) else enumerate(raw)
    for key, ex in iterator:
        q = str(ex.get("QUESTION", ex.get("question",""))).strip()
        ctx = ex.get("CONTEXTS", ex.get("contexts", []))
        if not isinstance(ctx, list): ctx = [str(ctx)]
        gold = str(ex.get("final_decision", ex.get("answer",""))).strip().lower()
        if gold not in {"yes","no","maybe"} or not q:
            bad += 1; continue
        items.append(YesNoMaybeItem(q, [str(c) for c in ctx], gold, str(key)))
    if bad: print(f"[WARN] [PubMedQA] skipped {bad} malformed item(s). Using {len(items)}.")
    return items

# ========================
# Prompt builders
# ========================
def apply_chat_template(user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def apply_chat_template_icl(demos: List[Tuple[str,str]], user_msg: str, system_msg: str = "") -> str:
    msgs = []
    if system_msg: msgs.append({"role":"system","content":system_msg})
    for du, da in demos:
        msgs.append({"role":"user","content":du})
        msgs.append({"role":"assistant","content":da})
    msgs.append({"role":"user","content":user_msg})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def mc_prompt(item: MCItem) -> str:
    letters = "".join(sorted(item.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in item.options.items()])
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

def mc_demo_user(it: MCItem) -> str:
    letters = "".join(sorted(it.options.keys()))
    opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
    return (
        "You are answering a multiple-choice medical question.\n"
        "Return ONLY one uppercase letter.\n\n"
        f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
        f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
    )

def mc_demo_assistant(it: MCItem) -> str:
    return it.answer_letter

def pubmedqa_demo_user(it: YesNoMaybeItem) -> str:
    ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
    return (
        "You are assessing a biomedical yes/no/maybe question.\n"
        "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
        f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
    )

def pubmedqa_demo_assistant(it: YesNoMaybeItem) -> str:
    return it.gold_label

# ========================
# Memory Bank (Memento-style)
# ========================
class MemoryBank:
    """Non-parametric store for solved cases + similarity retrieval."""
    def __init__(self, use_only_success: bool = True, capacity: int = 2000):
        self.use_only_success = use_only_success
        self.capacity = capacity
        self.cases: List[Dict[str, Any]] = []
        self._fit_ready = False

    def add(self, task_type: str, query_text: str, demo_user: str,
            demo_assistant: str, success: bool):
        if self.use_only_success and not success:
            return
        self.cases.append({
            "task_type": task_type,
            "query_text": (query_text or "").strip(),
            "demo_user": demo_user,
            "demo_assistant": demo_assistant,
        })
        if len(self.cases) > self.capacity:
            self.cases = self.cases[-self.capacity:]
        self._fit_ready = False

    def _ensure_index(self):
        texts = [c["query_text"] for c in self.cases]
        if _USE_EMB:
            self._emb = _SBERT.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 384))
        else:
            self._mat = _VEC.fit_transform(texts) if texts else None
        self._fit_ready = True

    def _similarities(self, query_text: str) -> np.ndarray:
        if not self.cases:
            return np.zeros((0,))
        if not self._fit_ready:
            self._ensure_index()
        if _USE_EMB:
            q = _SBERT.encode([query_text], normalize_embeddings=True)
            return (self._emb @ q[0]).astype(np.float32)
        else:
            q = _VEC.transform([query_text])
            return (self._mat @ q.T).toarray().ravel().astype(np.float32)

    def retrieve(self, task_type: str, query_text: str, top_k: int) -> List[Tuple[str, str]]:
        if not self.cases or top_k <= 0:
            return []
        sims = self._similarities(query_text)
        idx = np.argsort(-sims)
        demos: List[Tuple[str,str]] = []
        for i in idx:
            c = self.cases[i]
            if c["task_type"] != task_type:
                continue
            demos.append((c["demo_user"], c["demo_assistant"]))
            if len(demos) >= top_k:
                break
        return demos

CASE_BANK = MemoryBank(use_only_success=USE_ONLY_SUCCESS, capacity=MEM_CAPACITY)

# ========================
# Case Bank persistence + warm seed
# ========================
def save_case_bank(path: str = CASE_BANK_PATH):
    """Persist the in-memory case bank to JSON."""
    data = [{
        "task_type": c["task_type"],
        "query_text": c["query_text"],
        "demo_user": c["demo_user"],
        "demo_assistant": c["demo_assistant"],
    } for c in CASE_BANK.cases]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_case_bank(path: str = CASE_BANK_PATH):
    """Load a previously saved case bank JSON (if present)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    CASE_BANK.cases = list(data)
    CASE_BANK._fit_ready = False  # rebuild index lazily

def warm_seed_case_bank_mc(mc_items: List[MCItem], max_n: int = 8):
    """Preload a few strong MC demos (ideally from train/dev)."""
    for it in mc_items[:max_n]:
        CASE_BANK.add(
            task_type="mc",
            query_text=it.question,
            demo_user=mc_demo_user(it),
            demo_assistant=it.answer_letter,
            success=True,
        )

def warm_seed_case_bank_ynm(ynm_items: List[YesNoMaybeItem], max_n: int = 8):
    """Preload a few strong YNM demos (ideally from train/dev)."""
    for it in ynm_items[:max_n]:
        CASE_BANK.add(
            task_type="ynm",
            query_text=it.question,
            demo_user=pubmedqa_demo_user(it),
            demo_assistant=it.gold_label,
            success=True,
        )

# ========================
# ICL fallback sampling (when bank is short)
# ========================
rng = torch.Generator().manual_seed(RANDOM_SEED)

def sample_fewshot_mc(pool: List[MCItem], k: int, avoid_key: Optional[str]=None) -> List[MCItem]:
    if not pool or k <= 0: return []
    cand = [it for it in pool if it.source_id != avoid_key]
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[MCItem]] = {}
    for it in cand: by_label.setdefault(it.answer_letter, []).append(it)
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
    if not BALANCE_LABELS:
        idx = torch.randperm(len(cand), generator=rng).tolist()[:k]
        return [cand[i] for i in idx]
    by_label: Dict[str, List[YesNoMaybeItem]] = {}
    for it in cand: by_label.setdefault(it.gold_label, []).append(it)
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

# ========================
# Generation & parsing
# ========================
@torch.no_grad()
def generate_answer_with(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=DET_SAMPLE, pad_token_id=tokenizer.eos_token_id
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

def _retry_if_malformed(model, tokenizer, prompt: str,
                        allowed_letters: Optional[List[str]] = None,
                        ynm: bool = False):
    """Deterministic single retry on malformed output."""
    out = generate_answer_with(model, tokenizer, prompt)
    parsed = parse_ynm(out) if ynm else parse_mc_letter(out, allowed_letters or [])
    if parsed: return out, parsed
    stricter = prompt + "\nStrictly output only the final token. No explanations. Answer:"
    out2 = generate_answer_with(model, tokenizer, stricter)
    parsed2 = parse_ynm(out2) if ynm else parse_mc_letter(out2, allowed_letters or [])
    return (out2, parsed2) if parsed2 else (out, None)

# ========================
# Evaluation (MemoryBank retrieval + fallback + online writeback)
# ========================
def eval_mcq_with(model, tokenizer, items: List[MCItem], desc: str,
                  icl_pool: Optional[List[MCItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("mc", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_mc(icl_pool, need, avoid_key=it.source_id)
            demos += [(mc_demo_user(d), mc_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            letters = "".join(sorted(it.options.keys()))
            opts = "\n".join([f"{k}. {v}" for k,v in it.options.items()])
            user_msg = (
                "You are answering a multiple-choice medical question.\n"
                "Return ONLY one uppercase letter.\n\n"
                f"Question:\n{it.question}\n\nOptions:\n{opts}\n\n"
                f"Answer with ONLY ONE LETTER from [{letters}].\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = mc_prompt(it)
        allowed = sorted(list(it.options.keys()))
        out, pred = _retry_if_malformed(model, tokenizer, prompt, allowed_letters=allowed, ynm=False)
        used += 1
        is_correct = (pred or "") == it.answer_letter
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("mc", it.question, mc_demo_user(it), it.answer_letter, success=True)
    return correct / max(1, used)

def eval_pubmedqa_with(model, tokenizer, items: List[YesNoMaybeItem], desc: str,
                       icl_pool: Optional[List[YesNoMaybeItem]]=None, k_shots: int=0) -> float:
    correct, used = 0, 0
    for it in tqdm(items, desc=desc, ncols=80):
        demos = CASE_BANK.retrieve("ynm", it.question, top_k=k_shots)
        if len(demos) < k_shots and icl_pool and k_shots > 0:
            need = k_shots - len(demos)
            fb_items = sample_fewshot_ynm(icl_pool, need, avoid_key=it.source_id)
            demos += [(pubmedqa_demo_user(d), pubmedqa_demo_assistant(d)) for d in fb_items]
        if k_shots > 0 and demos:
            ctx = "\n".join(f"- {c}" for c in it.contexts[:6])
            user_msg = (
                "You are assessing a biomedical yes/no/maybe question.\n"
                "Return ONLY one token: yes, no, or maybe (lowercase).\n\n"
                f"Question:\n{it.question}\n\nEvidence:\n{ctx}\n\nAnswer:"
            )
            prompt = apply_chat_template_icl(demos, user_msg)
        else:
            prompt = pubmedqa_prompt(it)
        out, pred = _retry_if_malformed(model, tokenizer, prompt, ynm=True)
        used += 1
        is_correct = (pred or "") == it.gold_label
        correct += int(is_correct)
        if ENABLE_ONLINE_WRITEBACK and is_correct:
            CASE_BANK.add("ynm", it.question, pubmedqa_demo_user(it), it.gold_label, success=True)
    return correct / max(1, used)

# ========================
# Run helpers
# ========================
def run_all(tag: str, model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            k_mc: int, k_ynm: int):
    medqa_acc    = eval_mcq_with(model, tokenizer, medqa_items,   f"[{tag}] MedQA (k={k_mc})",
                                 icl_pool=medqa_pool,   k_shots=k_mc)
    medmcqa_acc  = eval_mcq_with(model, tokenizer, medmcqa_items, f"[{tag}] MedMCQA (k={k_mc})",
                                 icl_pool=medmcqa_pool, k_shots=k_mc)
    pubmedqa_acc = eval_pubmedqa_with(model, tokenizer, pubmedqa_items, f"[{tag}] PubMedQA (k={k_ynm})",
                                      icl_pool=pubmedqa_pool, k_shots=k_ynm)
    macro_acc = (medqa_acc + medmcqa_acc + pubmedqa_acc) / 3.0
    print(f"\n[{tag}] MedQA acc:    {medqa_acc:.3f}")
    print(f"[{tag}] MedMCQA acc:  {medmcqa_acc:.3f}")
    print(f"[{tag}] PubMedQA acc: {pubmedqa_acc:.3f}")
    print("-" * 52)
    print(f"[{tag}] Macro avg:    {macro_acc:.3f}")
    return medqa_acc, medmcqa_acc, pubmedqa_acc, macro_acc

# ========================
# CSV helper
# ========================
def append_csv_row(path: str, row: Dict[str, Any]):
    """Append one row of results to CSV, creating header if file does not exist."""
    file_exists = os.path.exists(path)
    fieldnames = [
        "tag",
        "model_repo",
        "adapter_repo",
        "k_shots_mc",
        "k_shots_ynm",
        "medqa_acc",
        "medmcqa_acc",
        "pubmedqa_acc",
        "macro_acc"
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ========================
# Main
# ========================
def main():
    # Load datasets
    medqa_items    = load_medqa(MEDQA_PATH)
    medmcqa_items  = load_medmcqa(MEDMCQA_PATH)
    pubmedqa_items = load_pubmedqa(PUBMEDQA_PATH)

    # Load existing case bank (if any), then warm seed a few demos
    load_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK loaded: {len(CASE_BANK.cases)} cases")

    # Warm seed (ideally use train/dev items; here we use first N for demo)
    warm_seed_case_bank_mc(medqa_items, max_n=WARM_SEED_MC_MAX)
    warm_seed_case_bank_ynm(pubmedqa_items, max_n=WARM_SEED_YNM_MAX)
    print(f"[INFO] CASE_BANK after warm seed: {len(CASE_BANK.cases)} cases")

    # Few-shot pools (fallback)
    medqa_pool, medmcqa_pool, pubmedqa_pool = medqa_items, medmcqa_items, pubmedqa_items

    # Baseline round
    base_medqa_acc, base_medmcqa_acc, base_pubmedqa_acc, base_macro_acc = run_all(
        "BASE", base_model, tokenizer,
        medqa_items, medmcqa_items, pubmedqa_items,
        medqa_pool, medmcqa_pool, pubmedqa_pool,
        K_SHOTS_MC, K_SHOTS_YNM
    )

    # Write BASE results to CSV
    append_csv_row(CSV_PATH, {
        "tag": "BASE",
        "model_repo": MODEL_REPO,
        "adapter_repo": "",
        "k_shots_mc": K_SHOTS_MC,
        "k_shots_ynm": K_SHOTS_YNM,
        "medqa_acc": base_medqa_acc,
        "medmcqa_acc": base_medmcqa_acc,
        "pubmedqa_acc": base_pubmedqa_acc,
        "macro_acc": base_macro_acc,
    })

    # Persist bank after baseline
    save_case_bank(CASE_BANK_PATH)
    print(f"[INFO] CASE_BANK saved (after BASE): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

    # Optional LoRA round
    if TEST_LORA_ADAPTER:
        print(f"[INFO] Loading LoRA adapter: {ADAPTER_REPO}")
        lora_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, torch_dtype=torch_dtype)
        lora_model.eval()
        lora_medqa_acc, lora_medmcqa_acc, lora_pubmedqa_acc, lora_macro_acc = run_all(
            "LoRA", lora_model, tokenizer,
            medqa_items, medmcqa_items, pubmedqa_items,
            medqa_pool, medmcqa_pool, pubmedqa_pool,
            K_SHOTS_MC, K_SHOTS_YNM
        )

        # Write LoRA results to CSV
        append_csv_row(CSV_PATH, {
            "tag": "LoRA",
            "model_repo": MODEL_REPO,
            "adapter_repo": ADAPTER_REPO,
            "k_shots_mc": K_SHOTS_MC,
            "k_shots_ynm": K_SHOTS_YNM,
            "medqa_acc": lora_medqa_acc,
            "medmcqa_acc": lora_medmcqa_acc,
            "pubmedqa_acc": lora_pubmedqa_acc,
            "macro_acc": lora_macro_acc,
        })

        save_case_bank(CASE_BANK_PATH)
        print(f"[INFO] CASE_BANK saved (after LoRA): {len(CASE_BANK.cases)} cases → {CASE_BANK_PATH}")

if __name__ == "__main__":
    main()

pip install ragas

import os

os.environ["OPENAI_API_KEY"] = "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"
