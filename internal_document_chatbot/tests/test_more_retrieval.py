# tests/test_more_retrieval.py
import os, sys, re, importlib, pytest

# Ensure project root on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Fast retrieval-only mode & mock data
os.environ.setdefault("SKIP_MODEL_LOAD", "1")
os.environ.setdefault("TRAIN_FILE", "tests/mock_dataset.jsonl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

inference = importlib.import_module("inference")
chat = inference.chat_with_model

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# 1) Camera-on mandatory — variations
@pytest.mark.parametrize("q", [
    "Are cameras required during meetings?",
    "Is video mandatory for meetings?",
    "Do I need to keep my camera on in meetings?",
    "Is camera-on mandatory for meetings?",
])
def test_camera_mandatory(q):
    got = chat(q)
    assert "camera" in norm(got) and "mandatory" in norm(got), f"\nQ: {q}\nGot: {got}"

# 2) Track/log/record/capture WFH hours — synonyms
@pytest.mark.parametrize("q", [
    "Where do I track my WFH hours?",
    "How do I log WFH hours?",
    "What tool is used to record Work From Home hours?",
    "What platform is used to track WFH hours?",
    "How do we capture remote work hours?",
])
def test_track_hours_synonyms(q):
    got = chat(q)
    ok = ("hrms" in norm(got)) and ("timesheet" in norm(got))
    assert ok, f"\nQ: {q}\nGot: {got}"

# 3) WFH support email — wording variants
@pytest.mark.parametrize("q", [
    "Which email should I use for WFH support?",
    "What mail to use for WFH support?",
    "What support mailbox handles WFH issues?",
    "What email to use for WFH support?",
])
def test_wfh_support_email(q):
    got = chat(q)
    assert "wfh-support@" in norm(got), f"\nQ: {q}\nGot: {got}"
