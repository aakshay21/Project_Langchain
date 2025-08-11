# Internal Document Chatbot

A Retrieval-First QA system trained on internal HR and company policy documents.
Uses RapidFuzz for fast matching, synonym/acronym expansion, and an optional LLM fallback for edge cases.

---

## Features
- **Fast retrieval** from a pre-loaded FAQ JSONL dataset.
- **Synonym & acronym expansion** for better matching (e.g., WFH → Work From Home).
- **Configurable thresholds** for strong/mid retrieval matches.
- **LLM fallback** (optional) for cases where no strong match is found.
- **CI-friendly mode** that skips model loading entirely.

---

## How It Works
1. **Dataset Load**: JSONL file with `### Question` and `### Answer` pairs.
2. **Preprocessing**: Expands acronyms and normalizes text.
3. **Retrieval**: Uses RapidFuzz for initial match, then re-ranks results with a weighted scoring function.
4. **Threshold Decision**: Returns dataset answer if above confidence cutoffs.
5. **Generation Fallback**: If no match passes threshold, builds a few-shot prompt and queries the LLM.

---

## Installation
```bash
git clone https://github.com/aakshay21/Project_Langchain.git
cd Project_Langchain/internal_document_chatbot
pip install -r requirements.txt

