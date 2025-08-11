# tests/test_retrieval.py
import os, sys, importlib, re
import pytest

# Ensure project root is importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force retrieval-only mode + point to mock dataset
os.environ["SKIP_MODEL_LOAD"] = "1"
os.environ["TRAIN_FILE"] = os.path.join(ROOT, "tests", "mock_dataset.jsonl")

inference = importlib.import_module("inference")
chat = inference.chat_with_model

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

def test_exact_questions():
    assert "camera" in norm(chat("Is camera-on mandatory for meetings?"))
    assert "hrms" in norm(chat("What platform is used to track WFH hours?"))
    assert "wfh-support@" in norm(chat("What email to use for WFH support?"))

@pytest.mark.parametrize("q,ok_substrings", [
    # camera-on variants
    ("Are cameras required during meetings?", ["camera", "mandatory"]),
    ("Is video mandatory for meetings?", ["camera", "mandatory"]),
    ("Do I need to keep my camera on in meetings?", ["camera", "mandatory"]),
    # track/log/record/capture WFH hours
    ("Where do I track my WFH hours?", ["hrms", "timesheet"]),
    ("How do I log WFH hours?", ["hrms", "timesheet"]),
    ("What tool is used to record Work From Home hours?", ["hrms", "timesheet"]),
    ("How do we capture remote work hours?", ["hrms", "timesheet"]),
    # support email wording
    ("Which email should I use for WFH support?", ["wfh-support@"]),
    ("What mail to use for WFH support?", ["wfh-support@"]),
    ("What support mailbox handles WFH issues?", ["wfh-support@"]),
])
def test_paraphrase_variants(q, ok_substrings):
    got = norm(chat(q))
    assert all(substr in got for substr in [s.lower() for s in ok_substrings]), f"\nQ: {q}\nGot: {got}"


