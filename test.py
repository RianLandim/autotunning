"""
Evaluate the base model and/or a saved adapter.
The test split is HUMAN ONLY — the agent may only run this with --split val
(e.g. `--base --split val` for the zero-shot reference during setup).

Usage:
    uv run test.py                        # adapter in out/adapter, test split
    uv run test.py --base                 # base model without adapter (zero-shot reference)
    uv run test.py --adapter PATH         # a specific adapter
    uv run test.py --base --split val     # zero-shot reference on val (allowed for the agent)
"""

import os
import argparse

from peft import PeftModel

from prepare import BASE_MODEL, OUTPUT_DIR, load_tokenizer, load_base_model, evaluate

parser = argparse.ArgumentParser(description="Evaluate the base model or an adapter")
parser.add_argument("--adapter", default=os.path.join(OUTPUT_DIR, "adapter"), help="Path to a saved LoRA adapter")
parser.add_argument("--base", action="store_true", help="Evaluate the base model without adapter")
parser.add_argument("--split", choices=["test", "val"], default="test", help="Split to evaluate on (test is human only)")
args = parser.parse_args()

tokenizer = load_tokenizer()
model = load_base_model()
if not args.base:
    model = PeftModel.from_pretrained(model, args.adapter)

results = evaluate(model, tokenizer, args.split, _allow_test=args.split == "test")
print("---")
print(f"model:            {BASE_MODEL}{'' if args.base else ' + ' + args.adapter}")
print(f"{args.split}_loss:{' ' * (12 - len(args.split))}{results['loss']:.6f}")
if "task_score" in results:
    print(f"{args.split}_task_score:{' ' * (6 - len(args.split))}{results['task_score']:.6f}")
