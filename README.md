# autotunning

A boilerplate for **autonomous fine-tuning research on any model**, based on Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). A coding agent (Claude Code, Codex, etc.) sets up the project with you, then runs experiments on its own: it edits the training code, trains for a fixed amount of time, measures the result, keeps or discards the change, and repeats.

The boilerplate ships with no predefined model or dataset. You pick both together with the agent during the configuration phase.

## How it works

The work happens in two phases, both described in `program.md`, which acts as the agent's "skill":

1. **Configuration (interactive, once per project).** The agent asks about the task, checks the GPU, proposes models that fit on it, converts your data to the expected format, defines the metric with you, measures the untrained model as a reference, and runs a quick training smoke test. You approve every choice. At the end, the configuration is committed and frozen.
2. **Experimentation (autonomous).** On an `autotunning/<tag>` branch, the agent only edits `train.py` and runs the loop indefinitely: train → measure → keep, or `git reset`. Results are logged to `results.tsv`.

## Structure

```
prepare.py      — task config, model loading, split, tokenization, evaluation (frozen after configuration)
train.py        — LoRA, optimizer, schedule, training loop (the agent edits this during experimentation)
program.md      — agent instructions (you edit this to change the "research method")
test.py         — base model / adapter evaluation; the test split is HUMAN ONLY
data/           — your dataset (data/dataset.jsonl) and converters, if any
examples/       — example task and end-to-end validation script
```

The **test split** is held out and the agent cannot access it. Since the agent runs hundreds of experiments against the validation split, it ends up fitting to it. The test split, run by you at the end, is the honest measure.

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
| `TASK_EVAL`, `PRIMARY_METRIC`, `score_prediction` | metric: `val_loss` (generic) or `task_score` (generates answers and scores them) |

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

Reference result on an RTX 3070 Ti (8 GB), with Qwen2.5-0.5B-Instruct and 2 minutes of training: test task_score **0.32 → 1.00**. Also use this command to check that the boilerplate still works after you change it.

## Using the result

Each `train.py` run saves the LoRA adapter to `out/adapter`. To reproduce the best experiment, check out the last kept commit of the `autotunning/<tag>` branch and run `uv run train.py`. Then:

```bash
uv run test.py --base     # base model on test (reference)
uv run test.py            # adapter on test
```

To serve the model, load the base with `PeftModel.from_pretrained(model, "out/adapter")`, or merge with `model.merge_and_unload()` and save the full model.

## Limitations

- One NVIDIA GPU, one process. No distributed training.
- Text models only (`AutoModelForCausalLM`). Multimodal or classification (encoder) models require adapting `prepare.py` and `train.py`.
- Reasoning models whose chat template rewrites history (e.g. strips `<think>`) don't work without adapting `apply_template`.

## License

MIT. Based on [autoresearch](https://github.com/karpathy/autoresearch) (MIT, Andrej Karpathy); see `LICENSE`.
