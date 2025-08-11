# batch_eval.py — sanity + regression run on your 34 Qs
import json, csv, os
from rapidfuzz import fuzz

# import your pipeline
from inference import chat_with_model, _extract_qa, _clean, TRAIN_FILE

# optional: quiet retrieval logs during batch run
# (or set DEBUG_RETRIEVAL=False in inference.py)
os.environ.setdefault("DEBUG_RETRIEVAL", "0")

IN = TRAIN_FILE  # reuse the same file inference.py points to
OUT = "batch_eval.csv"
THRESH = 85  # pass/fail threshold on fuzzy match

def main():
    rows = []
    total = 0
    passes = 0

    with open(IN, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            if not text and ("prompt" in obj and "completion" in obj):
                text = f"### Question: {obj['prompt']}\n\n### Answer: {obj['completion']}"

            q, a = _extract_qa(text)
            if not q or not a:
                continue

            expected = _clean(a)          # normalize ground-truth the same way
            predicted = chat_with_model(q)

            score = fuzz.QRatio(expected, predicted)
            ok = score >= THRESH

            rows.append({
                "id": i,
                "question": q,
                "expected": expected,
                "predicted": predicted,
                "fuzzy_score": score,
                "pass": ok
            })

            total += 1
            passes += int(ok)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","question","expected","predicted","fuzzy_score","pass"])
        w.writeheader(); w.writerows(rows)

    print(f"Evaluated {total} Qs — Passes: {passes}  Fails: {total - passes}  Accuracy: {passes/total:.1%}")
    print(f"Wrote details to {OUT}")

if __name__ == "__main__":
    main()
