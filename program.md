# autotunning

This is an experiment to have the LLM do its own fine-tuning research: adapt a pretrained model to one specific task as well as possible within a fixed compute budget.

There are two phases. **Phase 1 (Configuration)** is interactive: you set up the project together with the human, who makes the decisions. **Phase 2 (Experimentation)** is fully autonomous. If `BASE_MODEL` in `prepare.py` is already set and `.cache/meta.json` exists, the project is configured: skip to Phase 2's setup.

## Phase 1: Configuration (interactive, once per project)

Goal: define the task in `task.py`, fill in the configuration at the top of `prepare.py` and get a clean, prepared dataset. This is the only time `task.py` and `prepare.py` may be edited, and every choice below needs the human's approval.

1. **Define the task first (`task.py`).** Before anything else, ask the human what the model must do and what a correct answer looks like *in real use*, then fill in `task.py`:
   - `TASK_NAME` and `TASK_DESCRIPTION`: what comes in, what must come out, in the human's words.
   - `OUTPUT_TYPE`: `"json"` (structured extraction), `"label"` (classification) or `"text"` (free text).
   - The functional checks for that type: `JSON_FIELDS` with a comparator per field (`exact`, `normalized`, `number`, `set`, `contains`) and `JSON_STRICT_KEYS`; or `LABELS`; or `TEXT_REQUIRED_PATTERNS`.
   - `custom_checks()`: hard business rules a correct answer must satisfy (e.g. totals add up, dates are valid, ids exist in the input). Ask the human: "what would make an answer wrong even if it looks right?"
   - `PASS_THRESHOLD`: when an example counts as fully correct.
   Run `python task.py` (self-check) and test `check_prediction` on 2–3 real examples with the human (one correct answer, one wrong) to confirm the checks catch what matters.
2. **Read the in-scope files**: `README.md`, `task.py`, `prepare.py`, `train.py`, `test.py`, `benchmark.py`.
3. **Understand the data**: ask what data they have and how much. Keep it short — a few questions, not an interview.
4. **Check the hardware**: run `nvidia-smi` and note the GPU and VRAM.
5. **Choose the base model** with the human. Propose 1–3 candidates that fit the task and the VRAM (see the table in `README.md`; set `LOAD_IN_4BIT = True` for models that only fit in 4-bit). Requirements: a text causal LM loadable with `AutoModelForCausalLM`; an instruct/chat variant is usually best. Mention license and whether the model is gated (gated models need `hf auth login` by the human). Avoid reasoning models whose chat template rewrites history (e.g. strips `<think>`) unless you adapt `apply_template`. Set `BASE_MODEL` (and `TRUST_REMOTE_CODE` only if the model requires it and the human agrees).
6. **Prepare the data** in `data/dataset.jsonl`, one example per line:
   ```json
   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "target answer"}], "meta": {"category": "..."}}
   ```
   The loss is applied only to the last assistant message. If the human's data is in another format (CSV, spreadsheet, other JSON), write a converter in `data/` (e.g. `data/convert.py`) and run it — do not hand-edit records. Only create synthetic data if the human explicitly asks for it, and say clearly that results on synthetic data do not transfer automatically to real data. The optional `"meta"` object is ignored by training; put low-cardinality labels there (category, source, difficulty) so the benchmark can break results down by them.
7. **Golden set (strongly recommended)**: ask the human for real, human-verified cases that reflect production use, in the same format, saved as `data/benchmark.jsonl`. They must not be in `dataset.jsonl` (`benchmark.py` warns about overlap). Synthetic training data plus a real golden set is the honest way to know if tuning works in practice. The golden set is HUMAN ONLY, like the test split.
8. **Define the metric** with the human:
   - For `json` and `label` tasks (verifiable answers): set `TASK_EVAL = True` and `PRIMARY_METRIC = "task_score"` (it uses the checks in `task.py`), and set `MAX_NEW_TOKENS` to cover the longest expected answer.
   - For open-ended `text` without `TEXT_REQUIRED_PATTERNS`, keep `PRIMARY_METRIC = "val_loss"`.
9. **Set the budget**: `TIME_BUDGET` (default 300s; bigger models or datasets may need more) and `MAX_SEQ_LEN`.
10. **Prepare**: run `uv run prepare.py`. Check the printed lengths: if many examples are dropped or truncated, discuss raising `MAX_SEQ_LEN` with the human.
11. **Zero-shot reference**: run `uv run benchmark.py --only base --split val > base.log 2>&1` and report the base model's score, pass rate and failing checks to the human. This is what fine-tuning has to beat.
12. **Smoke test**: run `uv run train.py > run.log 2>&1` once. If it OOMs, adjust `DEVICE_BATCH_SIZE`/`GRAD_ACCUM_STEPS`/`GRADIENT_CHECKPOINTING` in `train.py` (or propose `LOAD_IN_4BIT` / a smaller model to the human) until it fits.
13. **Freeze**: make sure the project is a git repository (`git init` if needed) and commit the configuration (`task.py`, `prepare.py`, `data/`, `train.py`) on the main branch. From now on `task.py`, `prepare.py`, `benchmark.py` and `data/` are read-only.

## Phase 2 setup

To start a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct13`). The branch `autotunning/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotunning/<tag>` from the main branch.
3. **Read the in-scope files**: The project is small. Read these files for full context:
   - `README.md` — project context.
   - `task.py` — the task definition and its functional checks. Do not modify.
   - `prepare.py` — task configuration (base model, metric, time budget), data split, tokenization, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. LoRA config, optimizer, schedule, training loop.
   - A handful of lines of `data/dataset.jsonl` (e.g. `head -n 5`) to understand the task. Do not read the whole file.
4. **Verify data exists**: Check that `.cache/` contains `train.pt`, `val.pt` and `meta.json`. If not, run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first runs.
6. **Confirm and go**: Confirm setup looks good, and state the base model, the primary metric and its direction (from `PRIMARY_METRIC` in `prepare.py`).

Once you get confirmation, kick off the experimentation.

## Phase 2: Experimentation (autonomous)

Each experiment runs on a single GPU. The training script runs for a **fixed time budget** (`TIME_BUDGET` in `prepare.py`, wall clock training time, excluding model loading and evaluation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: LoRA rank/alpha/dropout, target modules, which layers are adapted, learning rate and schedule, batch size, gradient accumulation, optimizer, loss tweaks, gradient checkpointing, or even replacing LoRA with another parameter-efficient method or partial full fine-tuning, as long as it uses the packages already installed.

**What you CANNOT do:**
- Modify `task.py`, `prepare.py` or `benchmark.py`. They are read-only after Phase 1. They contain the task definition and its checks, the base model, the data split, the time budget and the evaluation harness.
- Modify or add data. Do not edit, generate, augment or read in bulk anything under `data/` (including `data/benchmark.jsonl`). Do not read `.cache/test.pt`.
- Evaluate on the test split or the golden set: never run `test.py` or `benchmark.py` without `--split val`, never pass `--data` to `benchmark.py`, and never call anything with `_allow_test=True`. They are reserved for the human's final check; touching them invalidates the whole run.
- Change the base model (`BASE_MODEL`) or tokenizer. The question is how well *this* model can be adapted.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation. The `evaluate` function in `prepare.py` is the ground truth metric.

**The goal: get the best primary metric on val.** If `PRIMARY_METRIC` is `task_score`, higher is better. If it is `val_loss`, lower is better. Since the time budget is fixed, you don't need to worry about training time. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a hard-ish constraint: the GPU may be small. If a run OOMs, reduce `DEVICE_BATCH_SIZE` (raising `GRAD_ACCUM_STEPS` to compensate) or enable `GRADIENT_CHECKPOINTING`. Peak VRAM also depends heavily on `MAX_SEQ_LEN` and the vocabulary size of the model (the logits tensor).

**Noise and overfitting**: The val split is small, so the metric is noisy. A change that "wins" by less than the noise level is not a real win. Fine-tuning datasets are also small, so watch `epochs` in the output: many epochs over few examples tends to overfit, and more training is not automatically better.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first runs**: Run the training script as is **twice** to establish the baseline and to measure noise. Record both. Use the spread between them as your noise estimate: only keep a change if it beats the best result by more than that spread (when results are equal within noise, prefer the simpler code).

## Output format

Once the script finishes it prints a summary like this:

```
---
val_loss:         0.123456
task_score:       0.850000
primary_metric:   task_score
training_seconds: 300.2
total_seconds:    341.7
peak_vram_mb:     5123.4
num_steps:        412
epochs:           12.45
trainable_params_M: 8.80
```

`task_score` is only printed when `TASK_EVAL` is enabled in `prepare.py`. You can extract the key metrics from the log file:

```
grep "^val_loss:\|^task_score:\|^peak_vram_mb:\|^epochs:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_loss	task_score	memory_gb	epochs	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss achieved (e.g. 0.123456) — use 0.000000 for crashes
3. task_score achieved — use `-` if not computed, 0.000000 for crashes
4. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
5. epochs, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_loss	task_score	memory_gb	epochs	status	description
a1b2c3d	0.081200	0.850000	5.0	12.4	keep	baseline
a1b2c3d	0.080900	0.840000	5.0	12.4	keep	baseline rerun (noise estimate)
b2c3d4e	0.074100	0.910000	5.1	12.6	keep	LORA_R 16 -> 32
c3d4e5f	0.090000	0.830000	5.0	12.4	discard	only attention modules
d4e5f6g	0.000000	0.000000	0.0	0.0	crash	batch size 16 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotunning/oct13`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_loss:\|^task_score:\|^peak_vram_mb:\|^epochs:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If the primary metric improved by more than the noise level, you "advance" the branch, keeping the git commit
9. If it is equal (within noise) or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Ideas to explore** (not exhaustive): learning rate (usually the highest-leverage knob), warmup/warmdown shape, LoRA rank and alpha (and their ratio), rsLoRA or DoRA (`use_rslora` / `use_dora` in `LoraConfig`), target modules (attention only vs. MLP too), adapting only the top N layers, LoRA dropout, weight decay, effective batch size, NEFTune-style embedding noise, label smoothing, fewer epochs vs. more epochs, freezing vs. training `lm_head`/embeddings, extra loss weight on the answer tokens that matter most (e.g. a label or decision token inside a longer answer).

**Timeout**: Each experiment should take about `TIME_BUDGET` plus model loading and evaluation. If a run exceeds 2x `TIME_BUDGET` + 5 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
