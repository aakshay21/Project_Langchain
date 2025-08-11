import os
from inference import chat_with_model, TRAIN_FILE

# Point to mock dataset for testing
os.environ["TRAIN_FILE"] = "tests/mock_dataset.jsonl"

TEST_CASES = [
    ("How should I apply for WFH?", "Submit the WFH request via HRMS portal at least 3 days in advance."),
    ("Can WFH be revoked?", "WFH can be revoked due to policy violations."),
    ("When can employees apply for long-term WFH?", "Apply for long-term WFH in April and November.")
]

def test_retrieval():
    for q, expected in TEST_CASES:
        ans = chat_with_model(q)
        assert expected.lower() in ans.lower(), f"FAIL for: {q} → got '{ans}'"

