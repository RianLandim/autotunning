"""
One-time data preparation + fixed evaluation for fine-tuning research.
Splits the task dataset into train/val/test and tokenizes it with the base
model's chat template.

Usage:
    uv run prepare.py

The task configuration below is filled in ONCE per project during setup
(by the human, or by the agent with the human's approval — see program.md).
During the experiment loop this file is read-only (the agent edits train.py).
"""

import os
import sys
import json
import random
import hashlib

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Task configuration (filled in during setup, then frozen)
# ---------------------------------------------------------------------------

BASE_MODEL = ""          # HF model id or local path of a text causal LM (chosen during setup)
TRUST_REMOTE_CODE = False  # only for models that require custom code from the Hub
LOAD_IN_4BIT = False     # QLoRA: load the frozen base model in 4-bit (models too big for the GPU in bf16)
DATA_PATH = "data/dataset.jsonl"  # one {"messages": [...]} per line
MAX_SEQ_LEN = 1024       # max tokens per example (prompt + answer)
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
VAL_FRACTION = 0.1       # fraction of examples used for validation
TEST_FRACTION = 0.1      # fraction held out for the final human-only test
SPLIT_SEED = 1337        # fixed seed so splits never change between runs

# Metrics. val_loss is always computed (lower is better). task_score generates
# answers for val examples and scores them with score_prediction (higher is better).
TASK_EVAL = False
PRIMARY_METRIC = "val_loss"  # "val_loss" or "task_score" (task_score requires TASK_EVAL)
EVAL_GEN_EXAMPLES = 100  # max val examples used for task_score
MAX_NEW_TOKENS = 128     # generation length for task_score
EVAL_BATCH_SIZE = 4      # batch size for val_loss
GEN_BATCH_SIZE = 16      # batch size for task_score generation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "out")
META_PATH = os.path.join(CACHE_DIR, "meta.json")
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Task scoring (customize per project, then frozen)
# ---------------------------------------------------------------------------

def score_prediction(prediction, reference):
    """
    Score one generated answer against the reference answer, in [0, 1].
    Default: JSON-aware exact match (compares parsed JSON when both sides parse,
    otherwise whitespace-stripped strings). Replace with whatever measures
    success for your use case (F1, label accuracy, regex checks, ...).
    """
    prediction, reference = prediction.strip(), reference.strip()
    try:
        return float(json.loads(prediction) == json.loads(reference))
    except (json.JSONDecodeError, TypeError):
        return float(prediction == reference)

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def check_config():
    if not BASE_MODEL:
        sys.exit("BASE_MODEL is not set. Choose a model and set it in prepare.py (see Phase 1 in program.md).")
    assert PRIMARY_METRIC in ("val_loss", "task_score")
    assert PRIMARY_METRIC != "task_score" or TASK_EVAL, "PRIMARY_METRIC=task_score requires TASK_EVAL=True"


def load_tokenizer():
    check_config()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=TRUST_REMOTE_CODE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    """Load the frozen base model on the GPU (bf16, fp16 on GPUs without bf16, or 4-bit if LOAD_IN_4BIT)."""
    check_config()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"dtype": dtype, "trust_remote_code": TRUST_REMOTE_CODE, "device_map": {"": 0}}
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)


# Plain format for tokenizers without a chat template (base / non-instruct models)
FALLBACK_ROLE_HEADERS = {"system": "### System:\n", "user": "### User:\n", "assistant": "### Assistant:\n"}


def apply_template(tokenizer, messages, add_generation_prompt=False):
    """Render messages with the model's chat template, or a plain fallback format if it has none."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = "".join(f"{FALLBACK_ROLE_HEADERS[m['role']]}{m['content']}{tokenizer.eos_token}\n\n" for m in messages)
    if add_generation_prompt:
        text += FALLBACK_ROLE_HEADERS["assistant"]
    return text


def data_fingerprint():
    with open(os.path.join(ROOT_DIR, DATA_PATH), "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    return {
        "base_model": BASE_MODEL,
        "max_seq_len": MAX_SEQ_LEN,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "data_sha256": data_hash,
    }


def read_dataset():
    """
    Read and validate the jsonl dataset. The last message of each example is the target.
    Returns the full records, so extra keys (e.g. metadata for analysis scripts) are kept.
    """
    records = []
    with open(os.path.join(ROOT_DIR, DATA_PATH), encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record["messages"]
            roles = [m["role"] for m in messages]
            assert len(messages) >= 2, f"line {lineno}: need at least one prompt message and one answer"
            assert roles[-1] == "assistant", f"line {lineno}: last message must be from the assistant"
            assert all(r in ("system", "user", "assistant") for r in roles), f"line {lineno}: invalid role in {roles}"
            records.append(record)
    return records


def split_dataset():
    """Deterministic train/val/test split of the raw records."""
    records = read_dataset()
    random.Random(SPLIT_SEED).shuffle(records)
    n_val = max(1, round(len(records) * VAL_FRACTION))
    n_test = max(1, round(len(records) * TEST_FRACTION))
    assert len(records) > n_val + n_test, f"Dataset too small ({len(records)} examples)"
    return {
        "test": records[:n_test],
        "val": records[n_test:n_test + n_val],
        "train": records[n_test + n_val:],
    }


def tokenize_example(tokenizer, messages):
    """
    Tokenize with the chat template. Loss is applied only to the final assistant
    message (prompt tokens get label -100). Returns None if the prompt alone
    does not fit in MAX_SEQ_LEN.
    """
    prompt_text = apply_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = apply_template(tokenizer, messages)
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template output for the full conversation does not start with the prompt "
            "(templates that rewrite history, e.g. reasoning models stripping <think>, are not supported as-is: "
            "pick a non-reasoning variant or adapt apply_template during setup)."
        )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= MAX_SEQ_LEN:
        return None
    completion_ids = completion_ids[:MAX_SEQ_LEN - len(prompt_ids)]
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "prompt_len": len(prompt_ids),
        "reference": messages[-1]["content"],
    }


def prepare():
    splits = split_dataset()

    print(f"Tokenizer: loading {BASE_MODEL}")
    tokenizer = load_tokenizer()
    os.makedirs(CACHE_DIR, exist_ok=True)
    for split, records in splits.items():
        tokenized = [tokenize_example(tokenizer, r["messages"]) for r in records]
        dropped = sum(1 for t in tokenized if t is None)
        tokenized = [t for t in tokenized if t is not None]
        truncated = sum(1 for t in tokenized if len(t["input_ids"]) == MAX_SEQ_LEN)
        lengths = sorted(len(t["input_ids"]) for t in tokenized)
        torch.save(tokenized, os.path.join(CACHE_DIR, f"{split}.pt"))
        print(f"  {split:5s}: {len(tokenized):6d} examples | median len {lengths[len(lengths) // 2]} | max len {lengths[-1]} | dropped {dropped} | truncated {truncated}")

    with open(META_PATH, "w") as f:
        json.dump(data_fingerprint(), f, indent=2)

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_split(split, _allow_test=False):
    """Load a tokenized split. The test split is reserved for the human (test.py)."""
    assert split in SPLITS
    assert split != "test" or _allow_test, "The test split is held out. Only test.py may read it."
    if not os.path.exists(META_PATH):
        sys.exit("No prepared data found. Run `uv run prepare.py` first.")
    with open(META_PATH) as f:
        if json.load(f) != data_fingerprint():
            sys.exit("Dataset or task config changed since the last prepare. Run `uv run prepare.py` again.")
    return torch.load(os.path.join(CACHE_DIR, f"{split}.pt"))


def collate(examples, pad_token_id, device="cuda"):
    """Right-pad a list of tokenized examples into (input_ids, attention_mask, labels)."""
    B = len(examples)
    T = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(e["labels"])
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def make_dataloader(tokenizer, batch_size, split="train", seed=0):
    """Infinite iterator of shuffled, padded batches. Yields (input_ids, attention_mask, labels, epoch)."""
    data = load_split(split)
    rng = random.Random(seed)
    epoch = 1
    while True:
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            batch = [data[j] for j in order[i:i + batch_size]]
            yield (*collate(batch, tokenizer.pad_token_id), epoch)
        epoch += 1

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE during experiments — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, tokenizer, split="val", _allow_test=False):
    """Mean cross-entropy per answer token (prompt tokens excluded)."""
    data = load_split(split, _allow_test)
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(data), EVAL_BATCH_SIZE):
        input_ids, attention_mask, labels = collate(data[i:i + EVAL_BATCH_SIZE], tokenizer.pad_token_id)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        targets = labels[:, 1:]
        mask = targets != -100
        total_loss += F.cross_entropy(logits[:, :-1][mask].float(), targets[mask], reduction="sum").item()
        total_tokens += mask.sum().item()
    return total_loss / total_tokens


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size=None):
    """Greedy-generate completions for a list of prompt token id lists (left-padded batches)."""
    batch_size = batch_size or GEN_BATCH_SIZE
    completions = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        T = max(len(p) for p in batch)
        input_ids = torch.full((len(batch), T), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), T), dtype=torch.long)
        for j, p in enumerate(batch):
            input_ids[j, T - len(p):] = torch.tensor(p)
            attention_mask[j, T - len(p):] = 1
        output = model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions += tokenizer.batch_decode(output[:, T:], skip_special_tokens=True)
    return completions


@torch.no_grad()
def evaluate_task(model, tokenizer, split="val", _allow_test=False):
    """Greedy-generate answers and average score_prediction over them."""
    data = load_split(split, _allow_test)[:EVAL_GEN_EXAMPLES]
    predictions = generate(model, tokenizer, [e["input_ids"][:e["prompt_len"]] for e in data])
    scores = [score_prediction(p, e["reference"]) for p, e in zip(predictions, data)]
    return sum(scores) / len(scores)


def evaluate(model, tokenizer, split="val", _allow_test=False):
    """Returns {"loss": ..., "task_score": ...} (task_score only if TASK_EVAL)."""
    was_training = model.training
    model.eval()
    results = {"loss": evaluate_loss(model, tokenizer, split, _allow_test)}
    if TASK_EVAL:
        results["task_score"] = evaluate_task(model, tokenizer, split, _allow_test)
    model.train(was_training)
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_config()
    if not os.path.exists(os.path.join(ROOT_DIR, DATA_PATH)):
        sys.exit(f"Dataset not found at {DATA_PATH}. See Phase 1 in program.md for the expected format.")
    print(f"Base model: {BASE_MODEL}")
    print(f"Dataset: {DATA_PATH}")
    prepare()
    print()
    print(f"Primary metric: {PRIMARY_METRIC} ({'higher' if PRIMARY_METRIC == 'task_score' else 'lower'} is better)")
    print("Done! Ready to train.")
