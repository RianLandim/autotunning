# Example: order extraction to JSON

A synthetic task used to validate the boilerplate end to end. Given customer messages in Portuguese written in varied ways, the model has to extract a JSON with `produto` (product), `quantidade` (quantity), `cidade` (city) and `pagamento` (payment). The quantity is sometimes spelled out ("duas", "doze") and payment is optional (`null`), so copying spans from the text is not enough.

```
Input:   "Me vê duas mochila no pix? É pra Recife."
Output:  {"produto": "mochila", "quantidade": 2, "cidade": "Recife", "pagamento": "pix"}
```

## How to run

From the repository root:

```bash
uv sync
uv run examples/run_example.py examples/order_extraction
```

The script copies `prepare.py`, `train.py` and `test.py` to `.example_runs/order_extraction/`, applies `config.json` to that copy (the root `prepare.py` is never modified) and runs:

1. `make_dataset.py`: generates 600 examples
2. `prepare.py`: 480/60/60 split and tokenization
3. `test.py --base`: the untrained base model, on the test split
4. `train.py`: LoRA fine-tuning for `TIME_BUDGET` seconds
5. `test.py`: the trained adapter, on the test split
6. `benchmark.py`: functional benchmark, base vs adapter on the test split (report in `bench/report.md`)

If fine-tuning doesn't beat the base model, or the benchmark verdict isn't `IMPROVED`, the script exits with an error. Logs for each step and a `results.json` are kept in the work folder.

## Configuration (`config.json`)

Top-level keys are applied to `prepare.py`; the `task` object is applied to `task.py`:

| Constant | Value |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `TIME_BUDGET` | 120 s |
| `MAX_SEQ_LEN` | 512 |
| `PRIMARY_METRIC` | `task_score` (per-field JSON checks from `task.py`) |
| `task.JSON_FIELDS` | `produto`/`cidade`: normalized, `quantidade`: number, `pagamento`: exact |
| `task.JSON_STRICT_KEYS` | `true` (extra keys fail the schema check) |

Each record also carries `meta.quantity_spelled_out` and `meta.has_payment`, so the benchmark report breaks the score down by those cases.

Hyperparameters are the `train.py` defaults (LoRA r=16 on all linear layers, LR 2e-4, 16 examples per step).

## Reference result

RTX 3070 Ti (8 GB), 2026-10-13:

| Test split (60 examples) | task_score | loss |
|---|---|---|
| Base model, untrained | 0.317 | 0.0990 |
| After fine-tuning | **1.000** | 0.0000 |

_Measured before the per-field scoring of `task.py` (all-or-nothing JSON match at the time). The base score will be higher with partial credit; re-run to update._

Training: 120 s, 303 steps (~10 epochs), 2.8 GB peak VRAM. Total script time: ~4 min, not counting the model download (~1 GB).

The task is easy on purpose: the goal is to prove the pipeline works (data → training → evaluation → measurable gain), not to probe the model's limits. On another GPU or library version the numbers may vary slightly, but fine-tuning should still land clearly above the base model.

## Trying the autonomous loop

To watch the agent work on the example, copy the work folder into its own repository and ask the agent for a round of experiments:

```bash
cp -r .example_runs/order_extraction ~/autotunning-example && cd ~/autotunning-example
cp ~/path/to/autotunning/{program.md,pyproject.toml,uv.lock,.gitignore} . && rm -f *.log results.json
git init && git add -A && git commit -m "example configuration"
```

Then, in the agent: *"Read program.md and let's start a new round of experiments."*
