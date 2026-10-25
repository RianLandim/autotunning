# autotunning

A boilerplate for **autonomous fine-tuning research on any model**, based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). A coding agent (Claude Code, Codex, etc.) sets up the project with you, then runs experiments on its own: it edits the training code, trains for a fixed amount of time, measures the result, keeps or discards the change, and repeats. At the end, a functional benchmark tells you, with statistics, whether the fine-tuned model really got better at your task.

The boilerplate ships with no predefined task, model or dataset. You define all three together with the agent during configuration, starting with the task.

## Quick start

```bash
git clone https://github.com/RianLandim/autotunning.git my-project && cd my-project
git remote remove origin   # the fine-tuning project becomes its own repo
uv sync
```

Then open the agent in that folder (ideally with permissions unlocked so it can run on its own) and follow the workflow:

| Step | Who | What to do |
|---|---|---|
| 1. Define the task | you + agent | Ask: *"Read program.md and let's set up a new fine-tuning project."* The agent first asks what the model must do and what a correct answer looks like, and writes it into `task.py`. |
| 2. Configure | you + agent | The agent checks the GPU, proposes base models, converts your data to `data/dataset.jsonl`, measures the untrained model and runs a smoke test. You approve every choice; the configuration is committed and frozen. |
| 3. Golden set | you | Put real, human-verified cases from production in `data/benchmark.jsonl` (same format, never used for training). Strongly recommended. |
| 4. Experiment | agent (autonomous) | Ask: *"Let's start a new round of experiments."* The agent loops on its own on an `autotunning/<tag>` branch, logging to `results.tsv`, until you stop it. |
| 5. Benchmark | you | `uv run benchmark.py --data data/benchmark.jsonl` (and/or `uv run benchmark.py` for the test split). Read `out/bench/<timestamp>/report.md`. |

The verdict at the top of the report (`IMPROVED`, `NO SIGNIFICANT DIFFERENCE` or `REGRESSED`) is the answer to "did tuning work?".

## How it works

Configuration and experimentation are described in `program.md`, which acts as the agent's "skill"; the benchmark is yours:

1. **Configuration (interactive, once per project).** The agent starts by **defining the task with you in `task.py`**: what goes in, what must come out, and the functional checks that decide if an answer is correct. Then it sets up the model, data, metric and time budget in `prepare.py`. Everything is committed and frozen.
2. **Experimentation (autonomous).** The agent only edits `train.py` and runs the loop indefinitely: train → measure on val → keep, or `git reset`. It optimizes `task_score`, which uses the same checks as the benchmark.
3. **Benchmark (human).** `benchmark.py` runs the base model and the fine-tuned adapter on the same held-out examples, checks every answer, and reports the gain with confidence intervals and a breakdown of what improved and what still fails.

The **test split** and the **golden set** are held out, and the agent is not allowed to evaluate on them. Since the agent runs hundreds of experiments against the validation split, it ends up fitting to it; the held-out sets, run by you at the end, are the honest measure.

## Structure

```
task.py         — task definition + functional checks (defined first, frozen after configuration)
prepare.py      — config, model loading, split, tokenization, evaluation (frozen after configuration)
train.py        — LoRA, optimizer, schedule, training loop (the agent edits this during experimentation)
program.md      — agent instructions (you edit this to change the "research method")
benchmark.py    — functional benchmark: base vs adapter, statistics, per-check/field/category report
test.py         — quick base model / adapter loss and task_score
data/           — dataset (dataset.jsonl), optional golden set (benchmark.jsonl), converters
examples/       — example task and end-to-end validation script
```

## Defining the task (`task.py`)

The first step of every project. The checks defined here are used both by `task_score` (what the agent optimizes) and by `benchmark.py` (what you use to judge the result), so the two cannot drift apart.

| Setting | What it is |
|---|---|
| `TASK_NAME`, `TASK_DESCRIPTION` | what the model is being specialized in, in real-use terms |
| `OUTPUT_TYPE` | `json` (structured extraction), `label` (classification) or `text` |
| `JSON_FIELDS`, `JSON_STRICT_KEYS` | per-field comparator: `exact`, `normalized` (case/accents/spaces), `number`, `set` (F1), `contains` |
| `LABELS` | allowed labels for classification |
| `TEXT_REQUIRED_PATTERNS` | regexes a correct free-text answer must match |
| `custom_checks()` | hard business rules (totals add up, valid dates, ids exist in the input); any failure scores 0 |
| `PASS_THRESHOLD` | score needed for an example to count as fully correct |

JSON answers get partial credit per field, plus checks for `valid_json`, `clean_output` (no markdown fence around it) and `schema`. A minimal JSON task looks like this:

```python
TASK_NAME = "invoice_extraction"
TASK_DESCRIPTION = "Given the text of an invoice, return a JSON with supplier, date (ISO), total and items."
OUTPUT_TYPE = "json"
JSON_FIELDS = {"supplier": "normalized", "date": "exact", "total": "number", "items": "set"}
JSON_STRICT_KEYS = True

def custom_checks(prediction, reference, record):
    data, _ = parse_json(prediction)
    return {"date_is_iso": bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data.get("date", ""))))}
```

Run `python task.py` for a self-check of the scoring logic (no GPU needed).

## Data format

`data/dataset.jsonl` (training) and `data/benchmark.jsonl` (golden set) use the same format, one example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "expected answer"}], "meta": {"source": "email", "difficulty": "hard"}}
```

The loss is applied only to the last assistant message. `meta` is optional and ignored by training; the benchmark breaks the score down by every low-cardinality key in it, so add the categories you care about.

## Benchmarking the result (`benchmark.py`)

```bash
uv run benchmark.py --data data/benchmark.jsonl   # base vs adapter on the golden set (recommended)
uv run benchmark.py                               # base vs out/adapter on the test split
uv run benchmark.py --only base                   # a single model (base | adapter)
uv run benchmark.py --adapter PATH --limit 200    # another adapter, fewer examples
uv run benchmark.py --split val                   # on val (the only mode the agent may use)
```

What you get, in `out/bench/<timestamp>/report.md` (plus `summary.json` and every prediction in `predictions.jsonl`):

- **Mean score and pass rate** for each model, with 95% bootstrap confidence intervals.
- **Paired comparison** on the same examples: delta (adapter − base) with a 95% CI, wins/losses/ties and a sign-test p-value. The verdict is `IMPROVED` / `REGRESSED` only when the whole CI is above / below zero; otherwise `NO SIGNIFICANT DIFFERENCE`.
- **Pass rate per functional check** (valid JSON, schema, custom rules) and **accuracy per field**, so you can see *what* got better.
- **Score per category** from each record's `meta` (e.g. `difficulty=hard`, `source=email`).
- **The worst failures** (input, expected, got), for error analysis.
- A warning if benchmark examples also appear in the training data (that measures memorization, not generalization).

Good scores on synthetic data do not prove the model works on real inputs; a golden set of real cases does.

## What gets configured (top of `prepare.py`)

| Constant | What it is |
|---|---|
| `BASE_MODEL` | Hugging Face ID or local path of a text causal LM |
| `LOAD_IN_4BIT` | QLoRA: loads the base model in 4-bit to fit smaller GPUs |
| `TRUST_REMOTE_CODE` | only for models that require custom code from the Hub |
| `DATA_PATH` | jsonl dataset in chat format (`{"messages": [...]}`) |
| `MAX_SEQ_LEN`, `TIME_BUDGET` | max example length and training time per experiment |
| `TASK_EVAL`, `PRIMARY_METRIC` | metric: `val_loss` (generic) or `task_score` (generates answers and scores them with the checks in `task.py`) |
| `MAX_NEW_TOKENS` | generation length for `task_score` and the benchmark |

VRAM reference for LoRA (approximate; also depends on `MAX_SEQ_LEN` and vocabulary size):

| VRAM  | bf16                     | 4-bit (`LOAD_IN_4BIT`) |
|-------|--------------------------|------------------------|
| 8 GB  | up to ~1.5B              | up to ~4B              |
| 16 GB | up to ~3–4B              | up to ~8B              |
| 24 GB | up to ~7–8B (with gradient checkpointing) | up to ~14B |
| 80 GB | up to ~14B+              | up to ~70B             |

Models without a chat template (base, non-instruct models) use a simple fallback format (`### User:` / `### Assistant:`).

## Example and validation

[`examples/order_extraction`](examples/order_extraction) is a small task that validates the boilerplate end to end: dataset → prepare → base model → training → adapter → benchmark. It runs in an isolated folder without modifying the root files, and fails unless fine-tuning beats the base model and the benchmark verdict is `IMPROVED`:

```bash
uv run examples/run_example.py examples/order_extraction
```

Its `config.json` shows how a configured project looks: top-level keys go to `prepare.py`, the `task` object goes to `task.py`. Use this command to check that the boilerplate still works after you change it.

Reference result on an RTX 3070 Ti (8 GB), with Qwen2.5-0.5B-Instruct and 2 minutes of training: test task_score **0.32 → 1.00**. That number was measured with the previous all-or-nothing JSON match; with the per-field scoring of `task.py` the base score will be higher and needs to be re-measured.

## Using the result

Each `train.py` run saves the LoRA adapter to `out/adapter`. To reproduce the best experiment, check out the last kept commit of the `autotunning/<tag>` branch and run `uv run train.py`. Then benchmark it (step 5 above).

To serve the model, load the base with `PeftModel.from_pretrained(model, "out/adapter")`, or merge with `model.merge_and_unload()` and save the full model.

## Limitations

- One NVIDIA GPU, one process. No distributed training.
- Text models only (`AutoModelForCausalLM`). Multimodal or classification (encoder) models require adapting `prepare.py` and `train.py`.
- Reasoning models whose chat template rewrites history (e.g. strips `<think>`) don't work without adapting `apply_template`.
- The benchmark measures the task defined in `task.py`. It does not measure general abilities; to check for catastrophic forgetting, run [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on general tasks with `--model_args pretrained=<base>,peft=out/adapter` for both models.

## License

MIT. Based on [autoresearch](https://github.com/karpathy/autoresearch) (MIT, Andrej Karpathy); see `LICENSE`.
