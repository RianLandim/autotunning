"""
Functional benchmark for the specialized task.

Generates answers with the base model and the fine-tuned adapter on the SAME
examples, checks every answer with the task's functional checks (task.py), and
reports, with statistics, whether the fine-tune really got better:

  - mean score and pass rate with 95% bootstrap confidence intervals
  - paired comparison base vs adapter: delta with 95% CI, wins/losses/ties, sign test p-value
  - pass rate of each functional check (valid JSON, schema, custom checks, ...)
  - accuracy per field (JSON tasks) and per category (any scalar key in a record's "meta")
  - the worst failures, so you can see what is still wrong

The test split and any golden set (--data) are HUMAN ONLY. The agent may only run
this with --split val.

Usage:
    uv run benchmark.py                          # base vs out/adapter on the test split
    uv run benchmark.py --data data/benchmark.jsonl   # base vs adapter on a golden set of real cases (recommended)
    uv run benchmark.py --split val              # on val (allowed for the agent)
    uv run benchmark.py --only base              # just one model (base | adapter)
    uv run benchmark.py --adapter PATH --limit 200

Reports are written to out/bench/<timestamp>/ (report.md, summary.json, predictions.jsonl).
"""

import os
import json
import math
import random
import hashlib
import argparse
from datetime import datetime

import task

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
WORST_FAILURES = 10

# ---------------------------------------------------------------------------
# Statistics (pure Python)
# ---------------------------------------------------------------------------


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Percentile bootstrap CI of the mean."""
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return (means[int(samples * alpha / 2)], means[min(samples - 1, int(samples * (1 - alpha / 2)))])


def sign_test_p(wins, losses):
    """Two-sided exact binomial sign test on non-tied pairs (H0: wins and losses equally likely)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(results):
    """Aggregate per-example check results of one model."""
    scores = [r["score"] for r in results]
    passed = [float(r["passed"]) for r in results]
    checks, fields = {}, {}
    for r in results:
        for name, ok in r["checks"].items():
            checks.setdefault(name, []).append(float(ok))
        for name, value in r["fields"].items():
            fields.setdefault(name, []).append(value)
    return {
        "n": len(results),
        "score": mean(scores), "score_ci95": bootstrap_ci(scores),
        "pass_rate": mean(passed), "pass_rate_ci95": bootstrap_ci(passed),
        "checks": {k: mean(v) for k, v in sorted(checks.items())},
        "fields": {k: mean(v) for k, v in fields.items()},
    }


def compare(base_results, tuned_results):
    """Paired comparison on the same examples."""
    deltas = [t["score"] - b["score"] for b, t in zip(base_results, tuned_results)]
    wins = sum(d > 0 for d in deltas)
    losses = sum(d < 0 for d in deltas)
    ci = bootstrap_ci(deltas)
    p = sign_test_p(wins, losses)
    if ci[0] > 0:
        verdict = "IMPROVED"
    elif ci[1] < 0:
        verdict = "REGRESSED"
    else:
        verdict = "NO SIGNIFICANT DIFFERENCE"
    return {"delta": mean(deltas), "delta_ci95": ci, "wins": wins, "losses": losses,
            "ties": len(deltas) - wins - losses, "sign_test_p": p, "verdict": verdict}


def categories(records, results_by_model, max_values=20):
    """Mean score per value of every scalar key found in record["meta"] (low-cardinality keys only)."""
    keys = {}
    for rec in records:
        for k, v in (rec.get("meta") or {}).items():
            if isinstance(v, (str, int, bool)) or v is None:
                keys.setdefault(k, set()).add(v)
    table = {}
    for k, values in keys.items():
        if len(values) > max_values:
            continue
        for v in sorted(values, key=str):
            idx = [i for i, rec in enumerate(records) if (rec.get("meta") or {}).get(k) == v]
            row: dict = {"n": len(idx)}
            for model, results in results_by_model.items():
                row[model] = mean([results[i]["score"] for i in idx])
            table[f"{k}={v}"] = row
    return table

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt_ci(ci, pct=False):
    return f"[{ci[0]:.1%}, {ci[1]:.1%}]" if pct else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def build_report(meta, summaries, comparison, cats, failures):
    models = list(summaries)
    lines = [f"# Benchmark: {task.TASK_NAME or '(task not named)'}", ""]
    if task.TASK_DESCRIPTION:
        lines += [f"> {task.TASK_DESCRIPTION}", ""]
    lines += [f"- **Base model:** `{meta['base_model']}`",
              f"- **Adapter:** `{meta['adapter']}`" if "adapter" in summaries else "- **Adapter:** not evaluated",
              f"- **Data:** {meta['source']} ({meta['n']} examples, {meta['skipped']} skipped for length)",
              f"- **Output type:** `{task.OUTPUT_TYPE}` · pass = score ≥ {task.PASS_THRESHOLD}",
              f"- **Date:** {meta['date']}", ""]
    if meta.get("overlap"):
        lines += [f"⚠️ **{meta['overlap']} benchmark examples also appear in the training split.** "
                  "Scores on them measure memorization, not generalization.", ""]

    lines += ["## Result", "", "| | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    lines.append("| Mean score | " + " | ".join(f"**{summaries[m]['score']:.3f}** {fmt_ci(summaries[m]['score_ci95'])}" for m in models) + " |")
    lines.append("| Pass rate | " + " | ".join(f"{summaries[m]['pass_rate']:.1%} {fmt_ci(summaries[m]['pass_rate_ci95'], pct=True)}" for m in models) + " |")
    lines.append("")
    if comparison:
        c = comparison
        lines += [f"**Verdict: {c['verdict']}.** Paired delta (adapter − base): **{c['delta']:+.3f}** "
                  f"95% CI {fmt_ci(c['delta_ci95'])} · wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']} "
                  f"· sign test p = {c['sign_test_p']:.4f}", "",
                  "_IMPROVED / REGRESSED only when the whole 95% CI of the delta is above / below zero._", ""]

    all_checks = sorted({k for m in models for k in summaries[m]["checks"]})
    if all_checks:
        lines += ["## Functional checks (pass rate)", "", "| Check | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k in all_checks:
            lines.append(f"| {k} | " + " | ".join(f"{summaries[m]['checks'].get(k, float('nan')):.1%}" for m in models) + " |")
        lines.append("")
    all_fields = list(dict.fromkeys(k for m in models for k in summaries[m]["fields"]))
    if all_fields:
        lines += ["## Accuracy per field", "", "| Field | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for k in all_fields:
            lines.append(f"| {k} | " + " | ".join(f"{summaries[m]['fields'].get(k, float('nan')):.1%}" for m in models) + " |")
        lines.append("")
    if cats:
        lines += ["## Score per category (from `meta`)", "", "| Category | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        for k, row in cats.items():
            lines.append(f"| {k} | {row['n']} | " + " | ".join(f"{row[m]:.3f}" for m in models) + " |")
        lines.append("")
    if failures:
        lines += [f"## Worst failures ({failures[0]['model']})", ""]
        for f in failures:
            failed = [k for k, ok in f["checks"].items() if not ok]
            failed += [f"field:{k}" for k, v in f.get("fields", {}).items() if v < 1]
            lines += [f"**#{f['index']}** score {f['score']:.2f}" + (f" · failed: {', '.join(failed)}" if failed else ""),
                      "", f"- input: `{f['input'][:300]}`", f"- expected: `{f['reference'][:300]}`",
                      f"- got: `{f['prediction'][:300]}`", ""]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Data and generation
# ---------------------------------------------------------------------------


def prompt_key(messages):
    return hashlib.sha256(json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                rec = json.loads(line)
                assert rec["messages"][-1]["role"] == "assistant", f"{path}:{lineno}: last message must be the assistant's answer"
                records.append(rec)
    return records


def last_user_message(messages):
    for m in reversed(messages[:-1]):
        if m["role"] == "user":
            return m["content"]
    return ""


def main():
    parser = argparse.ArgumentParser(description="Functional benchmark: base model vs fine-tuned adapter")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="dataset split (test is human only)")
    parser.add_argument("--data", help="separate golden set jsonl (same format as the dataset); human only")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (default: out/adapter)")
    parser.add_argument("--only", choices=["base", "adapter"], help="evaluate a single model")
    parser.add_argument("--limit", type=int, help="max examples")
    parser.add_argument("--out", help="output directory (default: out/bench/<timestamp>)")
    args = parser.parse_args()

    if not task.is_configured():
        raise SystemExit("task.py is not configured (TASK_NAME / TASK_DESCRIPTION). See Phase 1 in program.md.")

    import prepare  # heavy imports (torch, transformers) only when actually running
    from peft import PeftModel

    adapter = args.adapter or os.path.join(prepare.OUTPUT_DIR, "adapter")
    splits = prepare.split_dataset()
    if args.data:
        records, source = read_jsonl(args.data), args.data
    else:
        records, source = splits[args.split], f"{args.split} split"
    if args.limit:
        records = records[:args.limit]
    train_keys = {prompt_key(r["messages"]) for r in splits["train"]}
    overlap = sum(prompt_key(r["messages"]) in train_keys for r in records)

    tokenizer = prepare.load_tokenizer()
    items, skipped = [], 0
    for rec in records:
        tok = prepare.tokenize_example(tokenizer, rec["messages"])
        if tok is None:
            skipped += 1
            continue
        items.append((rec, tok["input_ids"][:tok["prompt_len"]]))
    records = [rec for rec, _ in items]
    prompts = [p for _, p in items]
    references = [rec["messages"][-1]["content"] for rec in records]
    print(f"Benchmark: {task.TASK_NAME} | {source} | {len(records)} examples ({skipped} skipped) | overlap with train: {overlap}")

    model = prepare.load_base_model()
    model.eval()
    predictions = {}
    if args.only != "adapter":
        print("Generating with the base model...")
        predictions["base"] = prepare.generate(model, tokenizer, prompts)
    if args.only != "base":
        print(f"Generating with the adapter ({adapter})...")
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        predictions["adapter"] = prepare.generate(model, tokenizer, prompts)

    results = {m: [task.check_prediction(p, ref, rec) for p, ref, rec in zip(preds, references, records)]
               for m, preds in predictions.items()}
    summaries = {m: summarize(r) for m, r in results.items()}
    comparison = compare(results["base"], results["adapter"]) if len(results) == 2 else None
    cats = categories(records, results)

    worst_model = "adapter" if "adapter" in results else "base"
    order = sorted(range(len(records)), key=lambda i: results[worst_model][i]["score"])
    failures = [{"model": worst_model, "index": i, "score": results[worst_model][i]["score"],
                 "checks": results[worst_model][i]["checks"], "fields": results[worst_model][i]["fields"],
                 "input": last_user_message(records[i]["messages"]),
                 "reference": references[i], "prediction": predictions[worst_model][i]}
                for i in order[:WORST_FAILURES] if not results[worst_model][i]["passed"]]

    meta = {"base_model": prepare.BASE_MODEL, "adapter": adapter, "source": source, "n": len(records),
            "skipped": skipped, "overlap": overlap, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
    out_dir = args.out or os.path.join(prepare.OUTPUT_DIR, "bench", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    report = build_report(meta, summaries, comparison, cats, failures)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"task": task.TASK_NAME, "meta": meta, "summaries": summaries, "comparison": comparison,
                   "categories": cats}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            row = {"index": i, "reference": references[i], "meta": rec.get("meta")}
            for m in predictions:
                row[m] = {"prediction": predictions[m][i], **results[m][i]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(report)
    print()
    print("---")
    summary_lines = []
    for m, s in summaries.items():
        summary_lines += [(f"bench_{m}_score", f"{s['score']:.6f}"), (f"bench_{m}_pass_rate", f"{s['pass_rate']:.6f}")]
    if comparison:
        summary_lines += [("bench_delta", f"{comparison['delta']:.6f}"),
                          ("bench_verdict", comparison["verdict"].replace(" ", "_"))]
    summary_lines.append(("bench_report", os.path.join(out_dir, "report.md")))
    for key, value in summary_lines:
        print(f"{key + ':':<26}{value}")


if __name__ == "__main__":
    main()
