"""
Task-specific fine-tuning script. Single-GPU, single-file.
LoRA on top of a pretrained causal LM, fixed wall-clock time budget.
Usage: uv run train.py
"""

import os
import gc
import math
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from prepare import BASE_MODEL, TIME_BUDGET, OUTPUT_DIR, PRIMARY_METRIC, load_tokenizer, load_base_model, load_split, make_dataloader, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"  # every linear layer except lm_head; or a list of module names

# Optimization
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Batching / memory
DEVICE_BATCH_SIZE = 4   # examples per forward/backward (reduce if OOM)
GRAD_ACCUM_STEPS = 4    # micro-batches per optimizer step
GRADIENT_CHECKPOINTING = False  # trades compute for memory

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

tokenizer = load_tokenizer()
model = load_base_model()
if GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
num_trainable, num_total = model.get_nb_trainable_parameters()
print(f"Base model: {BASE_MODEL}")
print(f"Trainable params: {num_trainable:,} / {num_total:,} ({100 * num_trainable / num_total:.2f}%)")

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, fused=True)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

num_train_examples = len(load_split("train"))
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, "train")
input_ids, attention_mask, labels, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Train examples: {num_train_examples:,}")
print(f"Examples per optimizer step: {DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}")

def compute_loss(model, input_ids, attention_mask, labels):
    """Mean cross-entropy over answer tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    targets = labels[:, 1:]
    mask = targets != -100
    return F.cross_entropy(logits[:, :-1][mask].float(), targets[mask])

# Schedule (based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
total_examples = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    for micro_step in range(GRAD_ACCUM_STEPS):
        loss = compute_loss(model, input_ids, attention_mask, labels)
        train_loss = loss.detach()
        (loss / GRAD_ACCUM_STEPS).backward()
        input_ids, attention_mask, labels, epoch = next(train_loader)
    torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    total_examples += DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 0:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up
    if step > 1 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
results = evaluate(model, tokenizer, "val")

# Save the adapter of this run (out/ is git-ignored; re-run the kept commit to reproduce)
model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_loss:         {results['loss']:.6f}")
if "task_score" in results:
    print(f"task_score:       {results['task_score']:.6f}")
print(f"primary_metric:   {PRIMARY_METRIC}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"epochs:           {total_examples / num_train_examples:.2f}")
print(f"trainable_params_M: {num_trainable / 1e6:.2f}")
