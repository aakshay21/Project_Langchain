import os
# Ensure retrieval-only tests never try to load the model
os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("TRAIN_FILE", "tests/mock_dataset.jsonl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import re
import pytest
import importlib

# Import after env is set
inference = importlib.import_module("inference")

# --- helpers ---
def ask(q: str) -> str:
    return inference.chat_with_model(q)

def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# --- exact retrieval tests (must hit dataset answers) ---
@pytest.mark.parametrize("q,exp", [
    ("Is camera-on mandatory for meetings?", "Cameras are mandatory during meetings."),
    ("What platform is used to track WFH hours?", "Track WFH hours using HRMS Timesheet."),
    ("Can WFH be revoked?", "WFH can be revoked due to policy violations."),
    ("When should WFH requests be submitted?", "Submit WFH requests at least 3 working days in advance."),
])
def test_exact_retrieval(q, exp):
    got = ask(q)
    assert norm(got) == norm(exp), f"\nQ: {q}\nExp: {exp}\nGot: {got}"

# --- small paraphrase subset (robustness, not strict equality) ---
PARA = {
    "Is camera-on mandatory for meetings?": [
        "Are cameras required during meetings?",
        "Do we have to keep the camera on in meetings?"
    ],
    "What platform is used to track WFH hours?": [
        "Which tool do we use to track WFH hours?",
        "Where should I log my WFH hours?"
    ],
    "Can WFH be revoked?": [
        "Is it possible that WFH gets revoked?",
        "Under what conditions can WFH be withdrawn?"
    ],
    "When should WFH requests be submitted?": [
        "By when do we need to submit WFH requests?",
        "What’s the submission window for WFH requests?"
    ],
}

@pytest.mark.parametrize("orig_q,paraphrases", list(PARA.items()))
def test_paraphrase_subset(orig_q, paraphrases):
    expected = ask(orig_q)
    for q in paraphrases:
        got = ask(q)
        # not strict string equality (allow synonyms); check loose containment/overlap
        assert any(tok in norm(got) for tok in norm(expected).split()[:3]), \
            f"\nOrig: {orig_q}\nExp: {expected}\nQ': {q}\nGot: {got}"
