"""
Task definition: WHAT the model is being specialized in and HOW a prediction
is checked. This is the first thing filled in during setup (Phase 1, step 1 in
program.md), then frozen together with prepare.py.

The same checks are used by:
  - prepare.py  -> task_score, the metric the agent optimizes during experiments
  - benchmark.py -> the functional benchmark you use to judge the final result
So the number being optimized is the number used to decide if tuning worked.

Pure Python (no torch), so it can be unit-tested anywhere:
    python task.py   # runs the self-check at the bottom
"""

import re
import json
import unicodedata

# ---------------------------------------------------------------------------
# Task definition (filled in during setup, then frozen)
# ---------------------------------------------------------------------------

TASK_NAME = ""            # short id, e.g. "order_extraction"
TASK_DESCRIPTION = ""     # 1-2 sentences in real-use terms: what comes in, what must come out
OUTPUT_TYPE = "json"      # "json" (structured extraction), "label" (classification) or "text" (free text)

# OUTPUT_TYPE = "json" -----------------------------------------------------
# Expected fields and how each one is compared. Comparators:
#   "exact"       identical value after JSON parsing
#   "normalized"  strings compared ignoring case, accents and extra whitespace
#   "number"      numeric equality (2 == 2.0 == "2"), within NUMBER_TOLERANCE
#   "set"         lists compared as sets (order-insensitive), scored by F1
#   "contains"    the reference string appears inside the predicted string (normalized)
# Empty dict = use every key of the reference with "exact".
JSON_FIELDS = {}
JSON_STRICT_KEYS = False  # True: extra keys in the prediction fail the "schema" check
NUMBER_TOLERANCE = 0.0    # absolute tolerance for "number" fields

# OUTPUT_TYPE = "label" ----------------------------------------------------
LABELS = []               # allowed labels; the prediction must be one of them (normalized)

# OUTPUT_TYPE = "text" -----------------------------------------------------
# Functional requirements a correct answer must satisfy (regexes, case-insensitive).
# Score = fraction of patterns matched. Empty list = normalized exact match.
TEXT_REQUIRED_PATTERNS = []

# An example "passes" when its score >= PASS_THRESHOLD (1.0 = fully correct).
PASS_THRESHOLD = 1.0


def custom_checks(prediction, reference, record):
    """
    Optional hard requirements specific to this task, as {check_name: bool}.
    Any failing custom check sets the example's score to 0.
    `prediction` is the raw model output; `parsed` JSON is available via parse_json(prediction).
    `record` is the full dataset record (may be None), including any "meta" keys.
    Examples: total == sum(items), date is ISO-8601, answer cites an id present in the input.
    """
    return {}

# ---------------------------------------------------------------------------
# Checking logic (generic; usually no need to edit)
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def is_configured():
    return bool(TASK_NAME and TASK_DESCRIPTION)


def normalize(value):
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().split())


def parse_json(text):
    """Parse a JSON answer, tolerating a surrounding ```json fence. Returns (value, clean) or raises ValueError."""
    text = text.strip()
    match = FENCE_RE.match(text)
    clean = match is None
    if match:
        text = match.group(1)
    try:
        return json.loads(text), clean
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(str(e)) from e


def _to_number(value):
    if isinstance(value, bool):
        raise ValueError("bool is not a number")
    return float(str(value).replace(",", ".")) if isinstance(value, str) else float(value)


def compare_field(pred, ref, comparator):
    """Score one field in [0, 1]."""
    if comparator == "exact":
        return float(pred == ref)
    if comparator == "normalized":
        if pred is None or ref is None:
            return float(pred is ref)
        return float(normalize(pred) == normalize(ref))
    if comparator == "number":
        if pred is None or ref is None:
            return float(pred is ref)
        try:
            return float(abs(_to_number(pred) - _to_number(ref)) <= NUMBER_TOLERANCE)
        except (TypeError, ValueError):
            return 0.0
    if comparator == "set":
        p = {normalize(x) for x in pred} if isinstance(pred, list) else set()
        r = {normalize(x) for x in ref} if isinstance(ref, list) else set()
        if not p and not r:
            return 1.0
        tp = len(p & r)
        return 0.0 if tp == 0 else 2 * tp / (len(p) + len(r))
    if comparator == "contains":
        if pred is None or ref is None:
            return float(pred is ref)
        return float(normalize(ref) in normalize(pred))
    raise ValueError(f"unknown comparator {comparator!r}")


def _check_json(prediction, reference):
    checks, fields = {}, {}
    ref = json.loads(reference)
    spec = JSON_FIELDS or {k: "exact" for k in (ref if isinstance(ref, dict) else {})}
    try:
        pred, clean = parse_json(prediction)
        checks["valid_json"] = True
        checks["clean_output"] = clean
    except ValueError:
        checks["valid_json"] = False
        checks["clean_output"] = False
        checks["schema"] = False
        return 0.0, checks, {name: 0.0 for name in spec}
    if not isinstance(ref, dict):  # non-object JSON answers: whole-value comparison
        checks["schema"] = type(pred) is type(ref)
        return float(pred == ref), checks, {}
    if not isinstance(pred, dict):
        checks["schema"] = False
        return 0.0, checks, {name: 0.0 for name in spec}
    missing = [k for k in spec if k not in pred]
    extra = [k for k in pred if k not in spec]
    checks["schema"] = not missing and not (JSON_STRICT_KEYS and extra)
    for name, comparator in spec.items():
        fields[name] = compare_field(pred.get(name), ref.get(name), comparator) if name in pred else 0.0
    score = sum(fields.values()) / len(fields) if fields else float(pred == ref)
    return score, checks, fields


def _check_label(prediction, reference):
    allowed = {normalize(l) for l in LABELS}
    pred = normalize(prediction)
    checks = {"valid_label": pred in allowed if allowed else True}
    return float(pred == normalize(reference)), checks, {}


def _check_text(prediction, reference):
    if not TEXT_REQUIRED_PATTERNS:
        return float(normalize(prediction) == normalize(reference)), {}, {}
    checks = {f"pattern_{i}": bool(re.search(p, prediction, re.I)) for i, p in enumerate(TEXT_REQUIRED_PATTERNS)}
    return sum(checks.values()) / len(checks), checks, {}


def check_prediction(prediction, reference, record=None):
    """
    Full functional check of one prediction.
    Returns {"score": float in [0, 1], "passed": bool, "checks": {name: bool}, "fields": {name: float}}.
    """
    if OUTPUT_TYPE == "json":
        score, checks, fields = _check_json(prediction, reference)
    elif OUTPUT_TYPE == "label":
        score, checks, fields = _check_label(prediction, reference)
    elif OUTPUT_TYPE == "text":
        score, checks, fields = _check_text(prediction, reference)
    else:
        raise ValueError(f"unknown OUTPUT_TYPE {OUTPUT_TYPE!r}")
    custom = {f"custom:{k}": bool(v) for k, v in custom_checks(prediction, reference, record).items()}
    checks.update(custom)
    if custom and not all(custom.values()):
        score = 0.0
    return {"score": score, "passed": score >= PASS_THRESHOLD, "checks": checks, "fields": fields}


def score_prediction(prediction, reference):
    """Scalar score in [0, 1] (used by prepare.py for task_score)."""
    return check_prediction(prediction, reference)["score"]


if __name__ == "__main__":
    # Self-check of the generic logic (independent of the configured task).
    OUTPUT_TYPE, JSON_FIELDS = "json", {"produto": "normalized", "quantidade": "number", "tags": "set"}
    ref = '{"produto": "Caneca Térmica", "quantidade": 2, "tags": ["a", "b"]}'
    r = check_prediction('```json\n{"produto": "caneca termica", "quantidade": "2", "tags": ["b", "a"]}\n```', ref)
    assert r["score"] == 1.0 and r["checks"]["valid_json"] and not r["checks"]["clean_output"], r
    r = check_prediction('{"produto": "caneca", "quantidade": 3}', ref)
    assert r["score"] == 0.0 and not r["checks"]["schema"], r
    r = check_prediction("not json", ref)
    assert r["score"] == 0.0 and not r["checks"]["valid_json"], r
    OUTPUT_TYPE, LABELS = "label", ["positivo", "negativo"]
    assert check_prediction("Positivo", "positivo")["passed"]
    assert not check_prediction("talvez", "positivo")["checks"]["valid_label"]
    print("task.py self-check: OK")
