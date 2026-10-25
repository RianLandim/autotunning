# autotunning

A boilerplate for **autonomous fine-tuning research on any model**, based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). A coding agent (Claude Code, Codex, etc.) sets up the project with you, then runs experiments on its own: it edits the training code, trains for a fixed amount of time, measures the result, keeps or discards the change, and repeats.

The boilerplate ships with no predefined model or dataset. You pick both together with the agent during the configuration phase.

## How it works

The work happens in three steps. The first two are described in `program.md`, which acts as the agent's "skill":

1. **Configuration (interactive, once per project).** The agent starts by **defining the task with you in `task.py`**: what goes in, what must come out, and the functional checks that decide if an answer is correct. Then it checks the GPU, proposes models that fit on it, converts your data to the expected format, measures the untrained model as a reference, and runs a quick training smoke test. You approve every choice. At the end, the configuration is committed and frozen.
2. **Experimentation (autonomous).** On an `autotunning/<tag>` branch, the agent only edits `train.py` and runs the loop indefinitely: train → measure → keep, or `git reset`. Results are logged to `results.tsv`.
3. **Benchmark (you).** At the end, `benchmark.py` runs the base model and the fine-tuned adapter on the same held-out examples, checks every answer with the task's functional checks, and tells you, with confidence intervals, whether the fine-tune really improved.

## Structure

```
task.py         — task definition + functional checks (defined first, frozen after configuration)
prepare.py      — config, model loading, split, tokenization, evaluation (frozen after configuration)
train.py        — LoRA, optimizer, schedule, training loop (the agent edits this during experimentation)
program.md      — agent instructions (you edit this to change the "research method")
test.py         — quick base model / adapter metrics; the test split is HUMAN ONLY
benchmark.py    — functional benchmark: base vs adapter, statistics, per-check/field/category report
data/           — your dataset (data/dataset.jsonl), optional golden set (data/benchmark.jsonl), converters
examples/       — example task and end-to-end validation script
```

The **test split** is held out and the agent cannot access it. Since the agent runs hundreds of experiments against the validation split, it ends up fitting to it. The test split, run by you at the end, is the honest measure.

## Defining the task (`task.py`)

The first step of every project. The checks defined here are used both by `task_score` (what the agent optimizes) and by `benchmark.py` (what you use to judge the result), so they cannot drift apart.

| Setting | What it is |
|---|---|
| `TASK_NAME`, `TASK_DESCRIPTION` | what the model is being specialized in, in real-use terms |
| `OUTPUT_TYPE` | `json` (structured extraction), `label` (classification) or `text` |
| `JSON_FIELDS`, `JSON_STRICT_KEYS` | per-field comparator: `exact`, `normalized` (case/accents/spaces), `number`, `set` (F1), `contains` |
| `LABELS` | allowed labels for classification |
| `TEXT_REQUIRED_PATTERNS` | regexes a correct free-text answer must match |
| `custom_checks()` | hard business rules (totals add up, valid dates, ids exist in the input); any failure scores 0 |
| `PASS_THRESHOLD` | score needed for an example to count as fully correct |

JSON answers get partial credit per field, plus checks for `valid_json`, `clean_output` (no markdown fence around it) and `schema`. Run `python task.py` for a self-check of the scoring logic.

## Benchmarking the result (`benchmark.py`)

```bash
uv run benchmark.py                               # base vs out/adapter on the test split
uv run benchmark.py --data data/benchmark.jsonl   # base vs adapter on a golden set of real cases (recommended)
uv run benchmark.py --split val                   # on val (the only mode the agent may use)
uv run benchmark.py --only base                   # a single model
```

What you get, in `out/bench/<timestamp>/report.md` (plus `summary.json` and every prediction in `predictions.jsonl`):

- **Mean score and pass rate** for each model, with 95% bootstrap confidence intervals.
- **Paired comparison** on the same examples: delta (adapter − base) with a 95% CI, wins/losses/ties and a sign-test p-value. The verdict is `IMPROVED` / `REGRESSED` only when the whole CI is above / below zero; otherwise `NO SIGNIFICANT DIFFERENCE`.
- **Pass rate per functional check** (valid JSON, schema, custom rules) and **accuracy per field**, so you can see *what* got better.
- **Score per category** for every low-cardinality key in each record's `"meta"` (e.g. `difficulty=hard`, `source=email`).
- **The worst failures** (input, expected, got), for error analysis.
- A warning if benchmark examples also appear in the training data (memorization, not generalization).

The strongest setup is a **golden set**: real, human-verified cases from production in `data/benchmark.jsonl`, never used for training. Good scores on synthetic data do not prove the model works on real inputs; the golden set does.

## Starting a project

```bash
git clone git@github.com:RianLandim/autotunning.git my-project && cd my-project
git remote remove origin   # the fine-tuning project becomes its own repo
uv sync
```

Then open the agent in that folder (ideally with permissions unlocked so it can run on its own) and ask:

```
Read program.md and let's set up a new fine-tuning project.
```

Once configuration is done:

```
Let's start a new round of experiments.
```

## What gets configured (top of `prepare.py`)

| Constant | What it is |
|---|---|
| `BASE_MODEL` | Hugging Face ID or local path of a text causal LM |
| `LOAD_IN_4BIT` | QLoRA: loads the base model in 4-bit to fit smaller GPUs |
| `TRUST_REMOTE_CODE` | only for models that require custom code from the Hub |
| `DATA_PATH` | jsonl dataset in chat format (`{"messages": [...]}`) |
| `MAX_SEQ_LEN`, `TIME_BUDGET` | max example length and training time per experiment |
| `TASK_EVAL`, `PRIMARY_METRIC` | metric: `val_loss` (generic) or `task_score` (generates answers and scores them with the checks in `task.py`) |

VRAM reference for LoRA (approximate; also depends on `MAX_SEQ_LEN` and vocabulary size):

| VRAM  | bf16                     | 4-bit (`LOAD_IN_4BIT`) |
|-------|--------------------------|------------------------|
| 8 GB  | up to ~1.5B              | up to ~4B              |
| 16 GB | up to ~3–4B              | up to ~8B              |
| 24 GB | up to ~7–8B (with gradient checkpointing) | up to ~14B |
| 80 GB | up to ~14B+              | up to ~70B             |

Models without a chat template (base, non-instruct models) use a simple fallback format (`### User:` / `### Assistant:`).

## Example and validation

[`examples/order_extraction`](examples/order_extraction) is a small task that validates the boilerplate end to end. It runs in an isolated folder without modifying the root `prepare.py`:

```bash
uv run examples/run_example.py examples/order_extraction
```

Reference result on an RTX 3070 Ti (8 GB), with Qwen2.5-0.5B-Instruct and 2 minutes of training: test task_score **0.32 → 1.00**. This number was measured with the previous all-or-nothing JSON match; the example now uses per-field scoring (`task.py`), so the base model score will be higher and needs to be re-measured. The script also runs `benchmark.py` and fails unless the verdict is `IMPROVED`. Also use this command to check that the boilerplate still works after you change it.

## Using the result

Each `train.py` run saves the LoRA adapter to `out/adapter`. To reproduce the best experiment, check out the last kept commit of the `autotunning/<tag>` branch and run `uv run train.py`. Then:

```bash
uv run test.py --base     # base model on test (reference)
uv run test.py            # adapter on test
uv run benchmark.py       # full functional benchmark, base vs adapter, with statistics
```

To serve the model, load the base with `PeftModel.from_pretrained(model, "out/adapter")`, or merge with `model.merge_and_unload()` and save the full model.

## Limitations

- One NVIDIA GPU, one process. No distributed training.
- Text models only (`AutoModelForCausalLM`). Multimodal or classification (encoder) models require adapting `prepare.py` and `train.py`.
- Reasoning models whose chat template rewrites history (e.g. strips `<think>`) don't work without adapting `apply_template`.

## License

MIT. Based on [autoresearch](https://github.com/karpathy/autoresearch) (MIT, Andrej Karpathy); see `LICENSE`.
