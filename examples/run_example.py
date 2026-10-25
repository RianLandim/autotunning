"""
End-to-end validation of the boilerplate on an example task.

Copies the boilerplate into an isolated work directory, applies the example's
config.json to the copy of prepare.py, generates the example dataset and runs
the full pipeline: prepare -> base model on test -> train -> adapter on test.
The repository's own prepare.py is never modified.

Exits with an error if the fine-tuned adapter does not beat the base model
on the primary metric.

Usage:
    uv run examples/run_example.py examples/order_extraction
"""

import os
import re
import sys
import json
import shutil
import argparse
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOILERPLATE_FILES = ["task.py", "prepare.py", "train.py", "test.py", "benchmark.py"]


def apply_config(path, config):
    """Overwrite top-level constants (NAME = value) in a boilerplate file (prepare.py or task.py)."""
    with open(path) as f:
        source = f.read()
    for name, value in config.items():
        pattern = rf"^{name} = .*$"
        assert len(re.findall(pattern, source, flags=re.M)) == 1, f"{name} not found exactly once in {os.path.basename(path)}"
        source = re.sub(pattern, lambda _: f"{name} = {value!r}", source, flags=re.M)
    with open(path, "w") as f:
        f.write(source)


def run(step, args, workdir):
    """Run a pipeline step, log it to <workdir>/<step>.log and return the parsed `key: value` summary."""
    log_path = os.path.join(workdir, f"{step}.log")
    print(f"[{step}] {' '.join(args)}", flush=True)
    with open(log_path, "w") as log:
        result = subprocess.run([sys.executable, *args], cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.exit(f"[{step}] failed (exit {result.returncode}), see {log_path}")
    with open(log_path) as f:
        return dict(re.findall(r"^(\w+):\s+(\S+)$", f.read(), flags=re.M))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the boilerplate end-to-end on an example task")
    parser.add_argument("example", help="Example directory (with config.json and make_dataset.py)")
    args = parser.parse_args()

    example_dir = os.path.abspath(args.example)
    name = os.path.basename(example_dir.rstrip("/"))
    with open(os.path.join(example_dir, "config.json")) as f:
        config = json.load(f)
    task_config = config.pop("task", {})

    workdir = os.path.join(ROOT_DIR, ".example_runs", name)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(os.path.join(workdir, "data"))
    for file in BOILERPLATE_FILES:
        shutil.copy(os.path.join(ROOT_DIR, file), workdir)
    apply_config(os.path.join(workdir, "task.py"), task_config)
    apply_config(os.path.join(workdir, "prepare.py"), config)
    print(f"Work dir: {workdir}")
    print(f"Config: {json.dumps(config)}")
    print(f"Task: {json.dumps(task_config, ensure_ascii=False)}")

    run("dataset", [os.path.join(example_dir, "make_dataset.py"), "data/dataset.jsonl"], workdir)
    run("prepare", ["prepare.py"], workdir)
    base = run("base", ["test.py", "--base"], workdir)
    train = run("train", ["train.py"], workdir)
    tuned = run("tuned", ["test.py"], workdir)
    bench = run("benchmark", ["benchmark.py", "--out", "bench"], workdir)

    metric = config.get("PRIMARY_METRIC", "val_loss")
    key = "test_task_score" if metric == "task_score" else "test_loss"
    base_value, tuned_value = float(base[key]), float(tuned[key])
    improved = tuned_value > base_value if metric == "task_score" else tuned_value < base_value

    results = {
        "example": name,
        "config": config,
        "base": {k: float(v) for k, v in base.items() if k.startswith("test_")},
        "tuned": {k: float(v) for k, v in tuned.items() if k.startswith("test_")},
        "train": {k: train[k] for k in ("training_seconds", "peak_vram_mb", "num_steps", "epochs") if k in train},
        "benchmark": {k: bench[k] for k in bench if k.startswith("bench_")},
        "improved": improved,
    }
    with open(os.path.join(workdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"| test split ({key}) | value |")
    print("|---|---|")
    print(f"| base model | {base_value:.4f} |")
    print(f"| fine-tuned | {tuned_value:.4f} |")
    print(f"Training: {train.get('num_steps')} steps, {train.get('epochs')} epochs, peak VRAM {train.get('peak_vram_mb')} MB")
    print(f"Benchmark: base {bench.get('bench_base_score')} -> adapter {bench.get('bench_adapter_score')} "
          f"({bench.get('bench_verdict')}), report: {os.path.join(workdir, 'bench', 'report.md')}")
    if not improved:
        sys.exit(f"FAIL: fine-tuning did not improve {key} over the base model")
    if bench.get("bench_verdict") != "IMPROVED":
        sys.exit("FAIL: the functional benchmark did not show a significant improvement")
    print("PASS: fine-tuning improved over the base model")
