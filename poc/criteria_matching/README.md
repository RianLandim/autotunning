# POC: criteria-based job/candidate matching with Qwen3-4B

Ranks candidates for a job and explains the ranking in natural language. The model never makes the decision directly. It does two narrow jobs, and deterministic code does the rest:

```
job description ──► [LLM] extract atomic requirements ──► requirements.json (human can review/edit)
                                                                 │
CV ──► [code] annotate period lengths ──► [LLM] for each (requirement, CV), independently:
                                               {justificativa, evidencia (literal quote), status}
                                                                 │
                    [code] verify the quote exists in the CV (else status = sem_evidencia)
                    [code] score = weighted sum (mandatory 3, nice-to-have 1)
                    [code] rank: all mandatory met first, then score
                    [code] explanation = the verdicts themselves, per candidate and per adjacent pair
```

Because the explanation is assembled from the same verdicts that produce the score, the stated reasons are the actual reasons. Every positive verdict cites a passage that is checked against the CV.

Model: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). It is the non-thinking instruct release of Qwen3-4B, so it has no `<think>` block to strip and gives direct JSON answers. No fine-tuning.

## Files

```
matcher.py              pipeline + CLI (prompts, evidence check, scoring rules, markdown report)
backends.py             LocalBackend (transformers, 4-bit) / SageMakerBackend (endpoint, OpenAI chat payload)
consistency_check.py    counterfactual test: re-evaluates every CV with only the name changed
deploy_sagemaker.py     creates/deletes the SageMaker endpoint (boto3)
samples/                one job, four CVs (fictional) and the human-reviewed requirements for the job
results/                outputs of the run documented below
```

## Running locally

```bash
uv run poc/criteria_matching/matcher.py \
    --job poc/criteria_matching/samples/job_backend_python.md \
    --candidates poc/criteria_matching/samples/candidates/*.md \
    --out out/poc
```

Outputs go to `out/poc/`:

- `report.md`: ranking, requirement matrix and explanations
- `ranking.json`: everything, for integration
- `requirements.json`: the extracted requirements

Review and edit `requirements.json` (wording, `tipo`, `peso`), then rerun with `--requirements <file>` to skip extraction. CVs are plain text/markdown with the candidate's name on the first line.

Locally the model runs in 4-bit (bitsandbytes) to fit 8 GB of VRAM. On SageMaker it runs in bf16, which is the same model at full precision, so quality there should be equal or better.

### Reproducing the results below

```bash
P=poc/criteria_matching
# 1. extraction (gives results/requirements_extracted.json)
uv run $P/matcher.py --job $P/samples/job_backend_python.md --candidates $P/samples/candidates/*.md --out out/extracted
# 2. ranking with the human-reviewed requirements (gives results/report.md and results/ranking.json)
uv run $P/matcher.py --requirements $P/samples/requirements_reviewed.json \
    --job $P/samples/job_backend_python.md --candidates $P/samples/candidates/*.md --out $P/results
# 3. name sensitivity (gives results/consistency_check.txt)
uv run $P/consistency_check.py --requirements $P/samples/requirements_reviewed.json --candidates $P/samples/candidates/*.md
```

Generation is greedy, so reruns on the same hardware and library versions give the same output. The extraction was identical across runs. Other GPUs or versions may change borderline verdicts.

## Results (RTX 3070 Ti 8 GB, 4-bit, 2026-10-13)

The test set is one job (backend Python) and four fictional CVs built to have clear differences.

### 1. Requirement extraction

[`results/requirements_extracted.json`](results/requirements_extracted.json) has 9 requirements: 6 mandatory and 3 nice-to-have. All of them are faithful to the job description, and the subjective traits ("proativo", "apaixonado por tecnologia") and the benefits were correctly left out.

One problem: it merged "Docker **and** CI/CD" into a single requirement. In the human review this was split in two, giving [`samples/requirements_reviewed.json`](samples/requirements_reviewed.json) (10 requirements). With the merged version, Diego (Docker yes, CI/CD no) got ✅ in one run and 🟡 in another for the same item.

### 2. Ranking with the reviewed requirements

Full report, with the requirement matrix, the reason each candidate is above the next and every cited passage: [`results/report.md`](results/report.md).

| # | Candidate | Profile | Score | Mandatory not met |
|---|---|---|---|---|
| 1 | Ana | Python backend 7+ years, fintech, AWS | 100.0 | — |
| 2 | Carla | Senior Java, occasional Python scripts, AWS, Kafka | 75.0 | Python 3+ years, FastAPI/Django |
| 3 | Diego | Data scientist, Python, 2 FastAPI services | 68.8 | CI/CD (partial), AWS |
| 4 | Bruno | Python/Flask 3 years, MySQL, no AWS | 37.5 | FastAPI/Django, relational DB ✗, CI/CD, AWS |

- **Correctness:** 40 verdicts. Checked by hand against the CVs, 37 are clearly right and 3 are debatable or wrong:
  - **Bruno R3 ❌ (wrong).** "Relational databases (PostgreSQL preferred)" against MySQL: MySQL is relational, so this should be ✅ or at least 🟡. The model read "preferred" as required.
  - **Diego R1 ✅ (debatable).** "3+ years of Python in backend development": he has 6y10m of Python as a data scientist, including 2 FastAPI services. Counting that as backend experience is a judgment call.
  - **Diego R5 🟡 (debatable).** "CI/CD pipelines": the model gave partial credit for Airflow + Docker, which is scheduling, not CI/CD. ❌ would be more accurate.
- **Evidence:** every ✅/🟡 cited a passage that exists in the CV, and no verdict was downgraded to ⚠️.
- **Ranking:** the order matches what a recruiter would expect for this job. Carla is ahead of Diego because she meets AWS and every nice-to-have, while her gap is the core Python/FastAPI skill. Whether that order is right depends on how the recruiter weighs the requirements (`peso`).
- **Time:** 69 s for 10 requirements × 4 candidates (40 calls), excluding model loading.

### 3. Name sensitivity

[`results/consistency_check.txt`](results/consistency_check.txt): each CV re-evaluated under six names suggesting different genders and origins (240 verdicts), with names **kept** in the text.

- **7/240 verdicts (2.9%) changed, all on borderline requirements.** Bruno's and Diego's CI/CD (Docker Compose / Airflow) changed in 6 cases, and Diego's "Python backend" in 1. Largest score change: 6.3 points.
- **Clear verdicts never changed.** Ana and Carla stayed the same under every name (0/120).
- **No direction by gender or origin.** For Bruno, "Maria", "Yasmin", "Hiroshi" and "Fatima" flipped, but "João" and "Kauã" didn't. For Diego, "João" and "Yasmin" flipped. The name acts as noise on verdicts that are already borderline. Part of this may be numerical: names of different lengths change batch padding, and greedy 4-bit decoding is sensitive to that.

Since no requirement needs the name, `matcher.py` **replaces it with "Candidato(a)"** before evaluation. `consistency_check.py` keeps names on purpose, to measure the raw model. This also explains why Bruno scores 37.5 in the ranking (anonymized: CI/CD ❌) and 43.8 in the check (real name: CI/CD 🟡). Other proxies remain in the text: gendered job titles ("Desenvolvedora"), city and school.

### What the iterations showed

Each problem below was found in an earlier run and fixed in code. They are the main lessons of the POC:

| Problem seen | Fix |
|---|---|
| Extraction skipped the "Diferenciais" section | Prompt: include nice-to-haves as `desejavel` |
| Wrong durations: "ago/2023 – atual" became "1 year", and Ana got "2 years" instead of 5y8m. The model doesn't know today's date and is bad at date math | **Code** computes each period's length and annotates the CV ("mar/2021 – atual [5 anos e 8 meses]"). The model only reads and sums the numbers |
| The verdict came before the reasoning | JSON key order is now `justificativa` → `evidencia` → `status`, so the model reasons first |
| Correct quotes spanning two bullets were rejected by the evidence check | Quotes are split into passages (`\|`, line breaks, a literal `\n`) and each passage is checked separately |
| Extraction merged two requirements into one | Human review of `requirements.json` before evaluation |

### Takeaways

- **The approach is viable for this use case without fine-tuning.**
  - Clear-cut verdicts are correct and did not change with names.
  - Explanations are faithful by construction.
  - Invented evidence is caught automatically.
- **The weak spots are borderline requirements (~7% of verdicts here) and requirement extraction.**
  - Borderline verdicts can change between runs or with irrelevant changes such as the name.
  - The one clear error (MySQL) is a nuance problem ("preferred" vs. "required").
  - Human review of `requirements.json` is the cheapest control. For example, rewording R3 as "relational database (e.g. PostgreSQL, MySQL)" removes the ambiguity.
- **Limits of this evidence:** this is 4 fictional CVs and 1 job, enough to validate the pipeline, not to estimate accuracy.
- **Next steps:**
  1. Run on real jobs and CVs, and have a recruiter label ~100 verdicts to measure accuracy per status.
  2. Detect borderline verdicts automatically: run each verdict 2–3× (or under 2 anonymized variants) and send disagreements to human review instead of scoring them.
  3. Compare 4-bit (local) with bf16 (SageMaker) on the same verdicts.
  4. If accuracy on real data is insufficient, the labeled verdicts become a fine-tuning dataset for the `autotunning` loop. The evaluation step is a causal LM with JSON output, which the boilerplate supports.

## Running on AWS (SageMaker)

### Architecture

```
app / backend ──► matcher.py (Lambda, ECS task or the app itself; CPU only)
                        │  boto3 invoke_endpoint, OpenAI chat payload, N requests in parallel
                        ▼
              SageMaker real-time endpoint
              ml.g6.xlarge (1× L4 24 GB) or ml.g5.xlarge (1× A10G 24 GB)
              AWS LMI container (vLLM) serving Qwen3-4B-Instruct-2507 in bf16
```

- **Only the LLM runs on the GPU.** The pipeline logic (`matcher.py`) is plain Python and runs wherever the application runs. It needs only `boto3` and IAM permission for `sagemaker:InvokeEndpoint`.
- **vLLM does continuous batching.** The requirement × candidate calls are independent, and `SageMakerBackend` sends them concurrently, so a job with 10 requirements × 20 candidates (200 calls) is processed as batches on the server.
- **Same code, two backends.** `--backend sagemaker --endpoint <name>` switches from the local GPU to the endpoint, so you can validate the endpoint with the same samples and compare its reports to the local ones.

### Deploy

Prerequisites: AWS credentials, a SageMaker execution role, a service quota for the instance type (`ml.g6.xlarge for endpoint usage`), and `uv sync --extra sagemaker`.

```bash
uv run --extra sagemaker poc/criteria_matching/deploy_sagemaker.py deploy \
    --name qwen3-4b-matcher --region us-east-1 \
    --role arn:aws:iam::<account>:role/<sagemaker-execution-role> \
    --image-uri <LMI image with vLLM >= 0.8.5>

uv run --extra sagemaker poc/criteria_matching/matcher.py --backend sagemaker \
    --endpoint qwen3-4b-matcher --region us-east-1 \
    --job poc/criteria_matching/samples/job_backend_python.md \
    --candidates poc/criteria_matching/samples/candidates/*.md --out out/poc_sagemaker

uv run --extra sagemaker poc/criteria_matching/deploy_sagemaker.py delete --name qwen3-4b-matcher --region us-east-1
```

**Container image:** Qwen3 needs vLLM 0.8.5 or newer. Pick the current LMI (`djl-inference`) tag from AWS's [available images list](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) and confirm its vLLM version. The `OPTION_*` variables in `deploy_sagemaker.py` follow the LMI v15 async vLLM configuration, so check them against the docs of the tag you pick. The Hugging Face vLLM DLC is an alternative, but its environment variables differ.

**Not validated yet.** The SageMaker path (deploy script and `SageMakerBackend`) was written but not run, because there are no AWS credentials in this environment and an endpoint costs money. Only the local path was executed.

### Cost and scaling options

| Option | When to use | Notes |
|---|---|---|
| Real-time endpoint, 1× `ml.g6.xlarge` | POC, steady traffic | ~US$1–1.5/h while it exists (region-dependent), **including idle time**. Delete it after tests. |
| Real-time + autoscaling 0→N (inference components) | Production with idle periods | Scale-to-zero is supported with inference components. Expect a cold start of minutes. |
| Asynchronous inference | Batch screening (e.g. rank everyone overnight) | Requests are queued in S3 and the endpoint can scale to 0 when the queue is empty. Fits screening better than real-time. |
| Batch transform | Large periodic batches | Pay only while the job runs. |
| Serverless inference | ✗ | No GPU support. |

Rough throughput: a 4B model in bf16 on an L4 with vLLM serves many parallel requests. Each verdict is ~1–1.5k prompt tokens and ~150 output tokens, so a few hundred verdicts per minute is a reasonable expectation. **Measure it on the endpoint** before sizing.

### Before production

- **Data protection (LGPD).** CVs are personal data. Keep the endpoint in the same region and account as the data, without public internet access (VPC). Don't log request bodies in plain text, and define retention for `ranking.json`.
- **Human review.** Output is a recommendation with evidence, not a decision. LGPD art. 20 gives candidates the right to review of automated decisions.
- **Requirements review.** The recruiter approves `requirements.json` (wording, mandatory vs. nice-to-have, weights) before candidates are evaluated. Most of the decision quality depends on this step.
- **Monitoring.** Track the share of `sem_evidencia` and `erro` verdicts per run, and run `consistency_check.py` periodically on real data.
- **PDF/DOCX CVs.** The POC takes plain text. Production needs an extraction step, such as Amazon Textract or a PDF-to-text library, upstream.
