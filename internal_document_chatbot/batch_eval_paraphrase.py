# batch_eval_paraphrase.py — paraphrase stress test + confusion summary
import os, re, json, csv, random
from collections import defaultdict, Counter
from rapidfuzz import fuzz, process

# import your pipeline + helpers + data
from inference import chat_with_model, _extract_qa, _clean, TRAIN_FILE, _FAQ_QS

random.seed(17)  # reproducible

# -------------------- config --------------------
PARAS_PER_Q   = 8      # paraphrases per original question
FUZZY_PASS    = 85     # expected vs predicted answer similarity threshold
NEAR_MISS_MIN = 70     # mark as "near miss" if score is in [NEAR_MISS_MIN, FUZZY_PASS)
OUT_CSV       = "paraphrase_eval.csv"

# ------------------- paraphraser -------------------
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
    "Could you tell me: {q}",
    "I need to know, {q}",
    "What’s the rule here — {q}",
    "As per company policy, {q}",
    "In simple words, {q}",
    "For employees, {q}",
    "Generally speaking, {q}",
    "Quick question: {q}",
]
REPHRASE_TEMPLATES = [
    lambda q: re.sub(r"^\s*what\s+is\s+the\s+", "Please share the ", q, flags=re.I),
    lambda q: re.sub(r"\?", "", q).strip() + "?",
    lambda q: "How does this work: " + q.rstrip("?") + "?",
    lambda q: re.sub(r"\bcan\b", "is it possible to", q, flags=re.I),
    lambda q: re.sub(r"\bwhen\b", "by what time", q, flags=re.I),
    lambda q: re.sub(r"\bwho\b", "which role", q, flags=re.I),
]

def _syn_swap_once(q: str) -> str:
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

    for _ in range(k*2):
        cands.add(_syn_swap_once(base))
    for _ in range(k):
        cands.add(random.choice(WRAP_TEMPLATES).format(q=_syn_swap_once(base)))
    for f in REPHRASE_TEMPLATES:
        try:
            cands.add(f(base))
        except Exception:
            pass

    cleaned = []
    for s in cands:
        s = re.sub(r"\s+", " ", s).strip()
        if not s.endswith("?"):
            s += "?"
        cleaned.append(s)

    cleaned = list(dict.fromkeys(cleaned))
    random.shuffle(cleaned)
    return cleaned[:k]

# ------------------- failure analysis -------------------
def categorize_failure(paraphrase: str, orig_q: str, expected: str, predicted: str, score: int):
    """
    Heuristics to label failure causes without modifying inference.py.
    """
    # 1) Guarded to OOD
    if predicted.strip() == "Not specified in the policy dataset.":
        # Did retrieval likely point to the right Q?
        guess = process.extractOne(paraphrase, _FAQ_QS, scorer=fuzz.QRatio)
        if guess:
            top_q, top_score, top_idx = guess
            if top_q != orig_q and top_score >= 80:
                return "guarded_not_specified_after_wrong_q"
        return "guarded_not_specified"

    # 2) Wrong dataset question was likely retrieved
    guess = process.extractOne(paraphrase, _FAQ_QS, scorer=fuzz.QRatio)
    if guess:
        top_q, top_score, top_idx = guess
        # Exact string match is best; allow tiny tolerance
        if top_q != orig_q and top_score - fuzz.QRatio(paraphrase, orig_q) > 2:
            return "wrong_dataset_match"

    # 3) Near miss vs low similarity
    if NEAR_MISS_MIN <= score < FUZZY_PASS:
        return "near_miss"
    return "low_similarity"

def summarize_failures(fails):
    """
    fails: list of dict rows for failures (from CSV rows we produce)
    Prints a summary and returns a dict of examples per category.
    """
    reasons = Counter([f["fail_reason"] for f in fails])
    print("\n=== Confusion Summary ===")
    if not reasons:
        print("No failures 🎉")
        return {}

    for reason, cnt in reasons.most_common():
        print(f"{reason}: {cnt}")

    # pick up to 2 examples per reason
    examples = defaultdict(list)
    for row in fails:
        r = row["fail_reason"]
        if len(examples[r]) < 2:
            examples[r].append({
                "paraphrase": row["paraphrase"],
                "expected": row["expected_answer"],
                "predicted": row["predicted_answer"],
                "score": row["fuzzy_score"],
                "orig_q": row["original_question"],
            })

    print("\n=== Examples per failure type (max 2 each) ===")
    for r, exs in examples.items():
        print(f"\n-- {r} --")
        for e in exs:
            print(f"Q': {e['paraphrase']}")
            print(f"orig: {e['orig_q']}")
            print(f"exp: {e['expected']}")
            print(f"got: {e['predicted']}")
            print(f"score: {e['score']}\n")
    return examples

# --------------------- main eval ---------------------
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

                row = {
                    "orig_id": i,
                    "para_id": f"{i}-{j}",
                    "original_question": q,
                    "paraphrase": pq,
                    "expected_answer": expected,
                    "predicted_answer": pred,
                    "fuzzy_score": int(score),
                    "pass": ok,
                    "fail_reason": ""
                }

                if not ok:
                    row["fail_reason"] = categorize_failure(pq, q, expected, pred, int(score))

                rows.append(row)
                total += 1
                passes += int(ok)

    # write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "orig_id","para_id","original_question","paraphrase",
            "expected_answer","predicted_answer","fuzzy_score","pass","fail_reason"
        ])
        w.writeheader()
        w.writerows(rows)

    acc = passes / total if total else 0.0
    print(f"\nParaphrase eval — Total: {total}  Passes: {passes}  Fails: {total-passes}  Accuracy: {acc:.1%}")
    print(f"Wrote details to {OUT_CSV}")

    # confusion summary
    fails = [r for r in rows if not r["pass"]]
    summarize_failures(fails)

if __name__ == "__main__":
    main()
