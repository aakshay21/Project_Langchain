# tests/test_eval.py
import os
import importlib
import re
import pytest

# --- Make tests fast & deterministic (retrieval-only) ---
os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("TRAIN_FILE", "tests/mock_dataset.jsonl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Import AFTER env is set so inference.py reads these
inference = importlib.import_module("inference")
chat = inference.chat_with_model

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# --- Sanity: dataset actually loaded ---
def test_dataset_loaded_minimum():
    # slightly hacky: indirectly verify by asking a known Q that must return non-empty
    ans = chat("How should I apply for WFH?")
    assert ans and ans != "Not specified in the policy dataset.", "Dataset not loaded or empty"

# --- Exact-match retrieval on mock items ---
@pytest.mark.parametrize("q,exp", [
    ("How should I apply for WFH?",
     "Submit the WFH request via HRMS portal at least 3 days in advance."),
    ("Can WFH be revoked?",
     "WFH can be revoked due to policy violations."),
    ("When can employees apply for long-term WFH?",
     "Apply for long-term WFH in April and November."),
])
def test_exact_retrieval(q, exp):
    got = chat(q)
    assert norm(exp) in norm(got), f"\nQ: {q}\nExp: {exp}\nGot: {got}"

# --- Paraphrase robustness (should still map to same answers) ---
PARA = {
    "How should I apply for WFH?": [
        "What is the process to apply for WFH?",
        "How do I submit a WFH request?",
        "Where do I apply for Work From Home?"
    ],
    "Can WFH be revoked?": [
        "Is it possible that WFH gets withdrawn?",
        "Under what conditions can Work From Home be revoked?",
        "Could WFH access be cancelled?"
    ],
    "When can employees apply for long-term WFH?": [
        "What months allow long-term WFH applications?",
        "When is the application window for long-term WFH?",
        "By what time can we apply for long duration WFH?"
    ],
}

@pytest.mark.parametrize("orig_q,paraphrases", list(PARA.items()))
def test_paraphrase_subset(orig_q, paraphrases):
    expected = chat(orig_q)
    for q in paraphrases:
        got = chat(q)
        # Allow light flexibility: first 3 tokens of expected should appear
        anchor = " ".join(norm(expected).split()[:3])
        assert anchor in norm(got), f"\nOrig: {orig_q}\nExp: {expected}\nQ': {q}\nGot: {got}"

# --- Edge case we saw earlier: awkward 'is it possible to WFH be revoked?' ---
def test_awkwark_revocation_phrase():
    q = "is it possible to WFH be revoked?"
    got = chat(q)
    # Accept either the exact sentence or something that clearly conveys revocation
    ok = (
        "revoked" in norm(got) and "wfh" in norm(got)
    ) or ("policy violations" in norm(got))
    assert ok, f"\nQ: {q}\nGot: {got}"
