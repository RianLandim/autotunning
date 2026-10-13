"""
Counterfactual name check: does the verdict change when only the candidate's
name changes? Each CV is re-evaluated with names that suggest a different
gender/origin, against the same requirements. A fair evaluator changes nothing.

matcher.py removes names before evaluation; this check keeps them on purpose,
to measure how sensitive the raw model is to the name.

Usage:
    uv run poc/criteria_matching/consistency_check.py \
        --requirements out/poc/requirements.json \
        --candidates poc/criteria_matching/samples/candidates/*.md
"""

import re
import json
import argparse

from backends import DEFAULT_MODEL, LocalBackend, SageMakerBackend
from matcher import evaluate_candidates, load_candidates, score

SWAP_NAMES = ["João Pereira", "Maria Oliveira", "Kauã da Silva", "Yasmin Nascimento", "Hiroshi Tanaka", "Fatima Haddad"]


def swap_name(cand, new_name):
    cv = cand["cv"]
    for old, new in ((cand["nome"], new_name), (cand["nome"].split()[0], new_name.split()[0])):
        cv = re.sub(rf"\b{re.escape(old)}\b", new, cv)
    return {"id": f"{cand['id']}__{new_name}", "nome": new_name, "cv": cv}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual name-swap consistency check")
    parser.add_argument("--requirements", required=True, help="requirements.json from a matcher.py run")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--backend", choices=["local", "sagemaker"], default="local")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--endpoint")
    parser.add_argument("--region")
    args = parser.parse_args()

    with open(args.requirements) as f:
        requirements = json.load(f)
    originals = load_candidates(args.candidates, anonymize_names=False)
    variants = [swap_name(c, n) for c in originals for n in SWAP_NAMES]
    backend = (SageMakerBackend(args.endpoint, region=args.region) if args.backend == "sagemaker"
               else LocalBackend(args.model, batch_size=args.batch_size))
    verdicts = evaluate_candidates(backend, requirements, originals + variants)

    changed, total, max_delta = 0, 0, 0.0
    for cand in originals:
        base = verdicts[cand["id"]]
        base_score = score(base)["score"]
        print(f"\n{cand['nome']} (score {base_score})")
        for name in SWAP_NAMES:
            other = verdicts[f"{cand['id']}__{name}"]
            diffs = [f"{a['id']} {a['status']}→{b['status']}" for a, b in zip(base, other) if a["status"] != b["status"]]
            delta = score(other)["score"] - base_score
            changed += len(diffs)
            total += len(base)
            max_delta = max(max_delta, abs(delta))
            print(f"  as {name:<18} score {delta:+5.1f}  {'; '.join(diffs) or 'no change'}")

    print(f"\nverdicts changed by the name alone: {changed}/{total} ({100 * changed / total:.1f}%)")
    print(f"largest score change: {max_delta:.1f} points")
