"""
Criteria-based job/candidate matching (POC).

The decision is not made by the model. The model only does two narrow jobs:

1. Extract atomic, verifiable requirements from the job description
   (saved to requirements.json so a human can review types/weights and rerun).
2. For each (requirement, candidate) pair, independently, say whether the CV
   meets it, quoting the CV passage that proves it.

Then deterministic code:
- checks that every quoted passage really exists in the CV (otherwise the
  verdict is downgraded to "sem_evidencia"),
- computes the score with explicit weights,
- ranks candidates and writes the explanation from the verdicts themselves,
  so the stated reasons are exactly what produced the score.

Usage:
    uv run poc/criteria_matching/matcher.py \
        --job poc/criteria_matching/samples/job_backend_python.md \
        --candidates poc/criteria_matching/samples/candidates/*.md \
        --out out/poc

    # against a SageMaker endpoint (see deploy_sagemaker.py)
    uv run --extra sagemaker poc/criteria_matching/matcher.py ... --backend sagemaker --endpoint qwen3-4b-matcher
"""

import os
import re
import json
import time
import argparse
import difflib
import datetime
import unicodedata

from backends import DEFAULT_MODEL, LocalBackend, SageMakerBackend

# ---------------------------------------------------------------------------
# Scoring rules (explicit, human-owned)
# ---------------------------------------------------------------------------

TYPE_WEIGHTS = {"obrigatorio": 3, "desejavel": 1}
STATUS_POINTS = {"atende": 1.0, "parcial": 0.5, "nao_atende": 0.0, "sem_evidencia": 0.0, "erro": 0.0}
STATUS_ICONS = {"atende": "✅", "parcial": "🟡", "nao_atende": "❌", "sem_evidencia": "⚠️", "erro": "💥"}
EVIDENCE_MIN_MATCH = 0.85  # fraction of the quote that must appear verbatim (after normalization) in the CV

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """Você é um analista de recrutamento. Sua tarefa é transformar a descrição de uma vaga em uma lista de requisitos verificáveis em um currículo.

Regras:
- Cada requisito deve ser atômico: uma única competência, tecnologia, experiência ou formação.
- Inclua apenas o que pode ser comprovado lendo um currículo. Ignore traços subjetivos (ex.: "proativo", "apaixonado por tecnologia") e benefícios da vaga.
- Nunca inclua características pessoais protegidas (idade, gênero, estado civil, origem, aparência, religião etc.).
- Preserve quantidades e níveis exigidos (ex.: "3+ anos", "inglês avançado").
- Inclua TODOS os requisitos da vaga, inclusive os listados como diferenciais, desejáveis ou "plus".
- "tipo" é "obrigatorio" se a vaga exige, ou "desejavel" se é diferencial, desejável, plus ou similar.

Responda somente com JSON no formato:
{"requisitos": [{"requisito": "...", "tipo": "obrigatorio"}]}"""

EVALUATE_SYSTEM = """Você avalia se um currículo atende a UM requisito de uma vaga.

Regras:
- Baseie-se somente no que está escrito no currículo. Não presuma nada que não esteja escrito.
- Ignore nome, gênero, idade, origem, foto ou qualquer característica pessoal: avalie apenas o requisito.
- "atende": o currículo comprova o requisito por completo (incluindo tempo ou nível exigido).
- "parcial": há evidência relacionada, mas incompleta (ex.: menos tempo que o exigido, tecnologia similar).
- "nao_atende": não há evidência no currículo.
- A duração de cada período já foi calculada e aparece entre colchetes, ex.: "[3 anos e 2 meses]". Use esses valores e some-os quando precisar; não recalcule a partir das datas.
- "justificativa": uma ou duas frases explicando a avaliação. Se o requisito pede tempo de experiência, mostre a conta com as datas.
- "evidencia": copie LITERALMENTE o trecho do currículo que sustenta a avaliação (sem reescrever). Se precisar de mais de um trecho, separe-os com " | ". Use "" se não houver.

Responda somente com JSON no formato:
{"justificativa": "...", "evidencia": "...", "status": "atende" | "parcial" | "nao_atende"}"""


def extract_requirements_prompt(job_text):
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"Descrição da vaga:\n\n{job_text}"}]


def evaluate_prompt(requirement, cv_text, today=None):
    today = today or datetime.date.today()
    return [{"role": "system", "content": f"{EVALUATE_SYSTEM}\n\nData de hoje: {today.strftime('%m/%Y')}"},
            {"role": "user", "content": f"Requisito: {requirement['requisito']}\n\nCurrículo:\n\n{cv_text}"}]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_json(text):
    """Parse the first JSON object in a model answer (tolerates code fences and surrounding text)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[*_`#>]", " ", text)  # markdown formatting the model may drop when quoting
    text = re.sub(r"(^|\n)\s*[-•]\s+", " ", text)  # list bullets
    return re.sub(r"\s+", " ", text).strip()


def evidence_found(quote, cv_text):
    """True if every passage of the quote (split on " | ", new lines or "...") appears verbatim in the CV."""
    cv = normalize(cv_text)
    passages = [normalize(p) for p in re.split(r"\s*\|\s*|\n|\\n|\.\.\.|…", quote)]
    passages = [p for p in passages if p]
    if not passages:
        return False
    for passage in passages:
        if passage in cv:
            continue
        match = difflib.SequenceMatcher(None, cv, passage, autojunk=False).find_longest_match(0, len(cv), 0, len(passage))
        if match.size / len(passage) < EVIDENCE_MIN_MATCH:
            return False
    return True


MONTHS = {m: i for i, m in enumerate(["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"], 1)}
DATE = r"(?:(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[a-zç]*\.?/)?((?:19|20)\d{2})"
PERIOD_RE = re.compile(rf"{DATE}\s*[–—-]\s*(?:{DATE}|(atual|presente|hoje))", re.I)


def annotate_durations(cv_text, today=None):
    """Append the computed length to every period ("mar/2021 – atual" -> "... [5 anos e 7 meses]").

    Small models get date arithmetic wrong, so durations are computed here instead of by the model.
    A period given only in years counts from January of the first year to December of the last."""
    today = today or datetime.date.today()

    def months(month, year, default):
        return int(year) * 12 + (MONTHS[month.lower()[:3]] if month else default)

    def replace(m):
        start = months(m.group(1), m.group(2), 1)
        end = today.year * 12 + today.month if m.group(5) else months(m.group(3), m.group(4), 12)
        total = end - start + 1
        if total <= 0:
            return m.group(0)
        years, rest = divmod(total, 12)
        parts = [f"{years} ano{'s' if years > 1 else ''}"] * bool(years) + [f"{rest} {'mês' if rest == 1 else 'meses'}"] * bool(rest)
        return f"{m.group(0)} [{' e '.join(parts)}]"

    return PERIOD_RE.sub(replace, cv_text)


def anonymize(cv_text, name):
    """Replace the candidate's full and first name: no requirement needs it, and it can only add noise or bias.
    Other proxies (gendered job titles, city, school) remain in the text."""
    for part in (name, name.split()[0]):
        cv_text = re.sub(rf"\b{re.escape(part)}\b", "Candidato(a)", cv_text)
    return cv_text


def candidate_name(cv_text, path):
    first_line = cv_text.strip().splitlines()[0]
    return first_line.lstrip("# ").strip() or os.path.splitext(os.path.basename(path))[0]

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def extract_requirements(backend, job_text):
    answer = backend.chat([extract_requirements_prompt(job_text)], max_new_tokens=1024)[0]
    requirements = parse_json(answer)["requisitos"]
    for i, req in enumerate(requirements, 1):
        req["id"] = f"R{i}"
        req["tipo"] = req.get("tipo") if req.get("tipo") in TYPE_WEIGHTS else "obrigatorio"
        req["peso"] = TYPE_WEIGHTS[req["tipo"]]
    return requirements


def evaluate_candidates(backend, requirements, candidates):
    """Evaluate every (requirement, candidate) pair independently. Returns {candidate_id: [verdict, ...]}."""
    pairs = [(c, r) for c in candidates for r in requirements]
    answers = backend.chat([evaluate_prompt(r, c["cv"]) for c, r in pairs], max_new_tokens=384)
    verdicts = {c["id"]: [] for c in candidates}
    for (cand, req), answer in zip(pairs, answers):
        verdict = {"id": req["id"], "requisito": req["requisito"], "tipo": req["tipo"], "peso": req["peso"]}
        try:
            parsed = parse_json(answer)
            status = parsed.get("status") if parsed.get("status") in ("atende", "parcial", "nao_atende") else "erro"
            verdict.update(status=status, evidencia=str(parsed.get("evidencia") or ""),
                           justificativa=str(parsed.get("justificativa") or ""))
        except (ValueError, json.JSONDecodeError):
            verdict.update(status="erro", evidencia="", justificativa=f"resposta inválida: {answer[:120]!r}")
        verdict["evidencia_verificada"] = evidence_found(verdict["evidencia"], cand["cv"])
        if verdict["status"] in ("atende", "parcial") and not verdict["evidencia_verificada"]:
            verdict["status_modelo"] = verdict["status"]
            verdict["status"] = "sem_evidencia"
        verdicts[cand["id"]].append(verdict)
    return verdicts


def score(verdicts):
    total = sum(v["peso"] for v in verdicts)
    points = sum(v["peso"] * STATUS_POINTS[v["status"]] for v in verdicts)
    missing = [v["id"] for v in verdicts if v["tipo"] == "obrigatorio" and v["status"] != "atende"]
    return {"score": round(100 * points / total, 1), "obrigatorios_pendentes": missing}


def rank(candidates, verdicts):
    results = []
    for cand in candidates:
        results.append({"id": cand["id"], "nome": cand["nome"], **score(verdicts[cand["id"]]),
                        "avaliacoes": verdicts[cand["id"]]})
    # candidates meeting every mandatory requirement first, then by score
    results.sort(key=lambda r: (len(r["obrigatorios_pendentes"]) > 0, len(r["obrigatorios_pendentes"]), -r["score"]))
    return results

# ---------------------------------------------------------------------------
# Explanations (built from the verdicts, not generated)
# ---------------------------------------------------------------------------


def describe(v):
    text = f"{v['requisito']} ({v['tipo']}): {v['justificativa']}"
    if v["evidencia"]:
        text += f' — evidência: "{v["evidencia"]}"'
    if v["status"] == "sem_evidencia":
        text += f" [o modelo disse '{v['status_modelo']}', mas o trecho citado não existe no currículo]"
    return text


def explain_candidate(result):
    lines = [f"### {result['nome']} — score {result['score']}"]
    if result["obrigatorios_pendentes"]:
        lines.append(f"Requisitos obrigatórios não comprovados: {', '.join(result['obrigatorios_pendentes'])}")
    for v in result["avaliacoes"]:
        lines.append(f"- {STATUS_ICONS[v['status']]} {v['id']} {describe(v)}")
    return "\n".join(lines)


def explain_pair(a, b):
    """Why `a` is ranked above `b`: the requirements where they differ, heaviest first."""
    diffs = []
    for va, vb in zip(a["avaliacoes"], b["avaliacoes"]):
        delta = va["peso"] * (STATUS_POINTS[va["status"]] - STATUS_POINTS[vb["status"]])
        if delta:
            diffs.append((delta, va, vb))
    lines = [f"### {a['nome']} ({a['score']}) acima de {b['nome']} ({b['score']})"]
    if not diffs:
        lines.append("Empate em todos os requisitos; a ordem entre os dois é arbitrária.")
    for delta, va, vb in sorted(diffs, key=lambda d: -abs(d[0])):
        ahead, behind = (a, b) if delta > 0 else (b, a)
        v_ahead, v_behind = (va, vb) if delta > 0 else (vb, va)
        lines.append(f"- {ahead['nome']} {STATUS_ICONS[v_ahead['status']]} vs. {behind['nome']} "
                     f"{STATUS_ICONS[v_behind['status']]} em **{va['requisito']}** ({va['tipo']})")
        lines.append(f"  - {ahead['nome']}: {v_ahead['justificativa']}")
        lines.append(f"  - {behind['nome']}: {v_behind['justificativa']}")
    return "\n".join(lines)


def markdown_report(job_name, requirements, ranking, backend_name, seconds):
    names = [r["nome"] for r in ranking]
    lines = [f"# Ranking — {job_name}", "",
             f"Backend: `{backend_name}` · {len(requirements)} requisitos × {len(ranking)} candidatos · {seconds:.0f}s", "",
             "| # | Candidato | Score | Obrigatórios pendentes |", "|---|---|---|---|"]
    for i, r in enumerate(ranking, 1):
        lines.append(f"| {i} | {r['nome']} | {r['score']} | {', '.join(r['obrigatorios_pendentes']) or '—'} |")
    lines += ["", "## Matriz de requisitos", "",
              "| Requisito | Tipo | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for i, req in enumerate(requirements):
        icons = " | ".join(STATUS_ICONS[r["avaliacoes"][i]["status"]] for r in ranking)
        lines.append(f"| {req['id']} {req['requisito']} | {req['tipo']} | {icons} |")
    lines += ["", "✅ atende · 🟡 parcial · ❌ não atende · ⚠️ modelo citou evidência inexistente · 💥 resposta inválida",
              "", "## Por que cada um ficou à frente do próximo", ""]
    lines += [explain_pair(a, b) + "\n" for a, b in zip(ranking, ranking[1:])]
    lines += ["## Avaliação detalhada", ""]
    lines += [explain_candidate(r) + "\n" for r in ranking]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_backend(args):
    if args.backend == "sagemaker":
        return SageMakerBackend(args.endpoint, region=args.region)
    return LocalBackend(args.model, load_in_4bit=not args.no_4bit, batch_size=args.batch_size)


def load_candidates(paths, anonymize_names=True):
    candidates = []
    for path in paths:
        with open(path) as f:
            cv = f.read()
        name = candidate_name(cv, path)
        cv = annotate_durations(cv)
        candidates.append({"id": os.path.splitext(os.path.basename(path))[0], "nome": name,
                           "cv": anonymize(cv, name) if anonymize_names else cv})
    return candidates


def run(backend, job_text, candidates, requirements=None):
    """Full pipeline. Returns (requirements, ranking)."""
    if requirements is None:
        requirements = extract_requirements(backend, job_text)
    verdicts = evaluate_candidates(backend, requirements, candidates)
    return requirements, rank(candidates, verdicts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Criteria-based job/candidate matching")
    parser.add_argument("--job", required=True, help="Job description (text/markdown file)")
    parser.add_argument("--candidates", nargs="+", required=True, help="CV files (text/markdown), first line = name")
    parser.add_argument("--requirements", help="Reuse a reviewed requirements.json instead of extracting it")
    parser.add_argument("--out", default="out/poc", help="Output directory")
    parser.add_argument("--backend", choices=["local", "sagemaker"], default="local")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="[local] HF model id or path")
    parser.add_argument("--no-4bit", action="store_true", help="[local] load in bf16 (needs ~10 GB VRAM for a 4B model)")
    parser.add_argument("--batch-size", type=int, default=4, help="[local] generation batch size")
    parser.add_argument("--endpoint", help="[sagemaker] endpoint name")
    parser.add_argument("--region", help="[sagemaker] AWS region")
    args = parser.parse_args()

    with open(args.job) as f:
        job_text = f.read()
    candidates = load_candidates(args.candidates)
    requirements = None
    if args.requirements:
        with open(args.requirements) as f:
            requirements = json.load(f)

    backend = make_backend(args)
    t0 = time.time()
    requirements, ranking = run(backend, job_text, candidates, requirements)
    seconds = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "requirements.json"), "w") as f:
        json.dump(requirements, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "ranking.json"), "w") as f:
        json.dump({"backend": backend.name, "seconds": round(seconds, 1), "requisitos": requirements,
                   "ranking": ranking}, f, ensure_ascii=False, indent=2)
    report = markdown_report(candidate_name(job_text, args.job), requirements, ranking, backend.name, seconds)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)

    print(report)
    print(f"\nSaved to {args.out}/ (requirements.json can be edited and reused with --requirements)")
