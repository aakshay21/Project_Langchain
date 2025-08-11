# paraphrase_eval.py — stress test retrieval with paraphrased questions (no internet)
import os, re, json, csv, random
from rapidfuzz import fuzz
from inference import chat_with_model, _extract_qa, _clean, TRAIN_FILE

random.seed(17)  # reproducible

# How many paraphrases per original question
PARAS_PER_Q = 8
FUZZY_PASS = 85  # expected vs predicted answer similarity threshold
OUT_CSV = "paraphrase_eval.csv"

# Light synonym pool and phrase variants (extend as you like)
SYN = {
    r"\bapply for\b": ["request", "submit a request for", "raise a request for", "seek"],
    r"\bapply\b": ["request", "seek", "raise a request"],
    r"\bwork from home\b": ["WFH", "remote work", "telework", "work remotely"],
    r"\bshort[- ]?term\b": ["temporary", "brief", "limited period"],
    r"\blong[- ]?term\b": ["extended", "long duration"],
    r"\bapprove\b": ["authorize", "greenlight", "okay"],
    r"\bapprover\b": ["authorizer", "manager approver"],
    r"\bdeadline\b": ["cutoff", "last date", "submission window"],
    r"\bVPNs?\b": ["VPN connection", "corporate VPN"],
    r"\bmandatory\b": ["compulsory", "required", "must"],
    r"\bpolicy\b": ["guideline", "rule"],
    r"\bemail\b": ["mail", "support mailbox"],
    r"\bHRMS\b": ["HR portal", "HR system"],
    r"\bdevice\b": ["laptop", "company machine"],
}

WRAP_TEMPLATES = [
    "Could you tell me: {q} ",
    "I need to know, {q}",
    "What’s the rule here — {q}",
    "As per company policy, {q}",
    "In simple words, {q}",
    "For employees, {q}",
    "Generally speaking, {q}",
    "Quick question: {q}",
]

# Basic question rewrites
REPHRASE_TEMPLATES = [
    lambda q: re.sub(r"^\s*what\s+is\s+the\s+", "Please share the ", q, flags=re.I),
    lambda q: re.sub(r"\?", "", q).strip() + "?",
    lambda q: "How does this work: " + q.rstrip("?") + "?",
    lambda q: re.sub(r"\bcan\b", "is it possible to", q, flags=re.I),
    lambda q: re.sub(r"\bwhen\b", "by what time", q, flags=re.I),
    lambda q: re.sub(r"\bwho\b", "which role", q, flags=re.I),
]

def _syn_swap_once(q: str) -> str:
    """Randomly pick 1–2 patterns to substitute."""
    patterns = list(SYN.keys())
    random.shuffle(patterns)
    n = random.choice([1,1,2])  # mostly 1 substitution
    out = q
    used = 0
    for pat in patterns:
        if used >= n:
            break
        if re.search(pat, out, flags=re.I):
            repl = random.choice(SYN[pat])
            out = re.sub(pat, repl, out, flags=re.I)
            used += 1
    return out

def generate_paraphrases(q: str, k: int) -> list:
    cands = set()
    base = q.strip()

    # direct synonyms
    for _ in range(k*2):
        cands.add(_syn_swap_once(base))

    # wrapping templates
    for _ in range(k):
        cands.add(random.choice(WRAP_TEMPLATES).format(q=_syn_swap_once(base)))

    # rule-based quick rewrites
    for f in REPHRASE_TEMPLATES:
        try:
            cands.add(f(base))
        except Exception:
            pass

    # ensure question mark
    cleaned = []
    for s in cands:
        s = re.sub(r"\s+", " ", s).strip()
        if not s.endswith("?"):
            s += "?"
        cleaned.append(s)

    # sample k unique paraphrases
    cleaned = list(dict.fromkeys(cleaned))  # preserve order, unique
    random.shuffle(cleaned)
    return cleaned[:k]

def main():
    rows = []
    total = 0
    passes = 0

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            if not text and ("prompt" in obj and "completion" in obj):
                text = f"### Question: {obj['prompt']}\n\n### Answer: {obj['completion']}"
            q, a = _extract_qa(text)
            if not q or not a:
                continue

            expected = _clean(a)
            paras = generate_paraphrases(q, PARAS_PER_Q)

            for j, pq in enumerate(paras, 1):
                pred = chat_with_model(pq)
                score = fuzz.QRatio(expected, pred)
                ok = score >= FUZZY_PASS

                rows.append({
                    "orig_id": i,
                    "para_id": f"{i}-{j}",
                    "original_question": q,
                    "paraphrase": pq,
                    "expected_answer": expected,
                    "predicted_answer": pred,
                    "fuzzy_score": score,
                    "pass": ok
                })
                total += 1
                passes += int(ok)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "orig_id","para_id","original_question","paraphrase",
            "expected_answer","predicted_answer","fuzzy_score","pass"
        ])
        w.writeheader()
        w.writerows(rows)

    acc = passes / total if total else 0.0
    print(f"Paraphrase eval — Total: {total}  Passes: {passes}  Fails: {total-passes}  Accuracy: {acc:.1%}")
    print(f"Wrote details to {OUT_CSV}")

if __name__ == "__main__":
    main()
