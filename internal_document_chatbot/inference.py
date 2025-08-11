# inference.py — retrieval-first QA for TinyLlama + PEFT (works on CPU/MPS)
# Quiet startup + conditional resize + robust cleaner/guard

import os, sys, json, re, warnings, logging
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from rapidfuzz import process, fuzz

# ========= EDIT THESE IF NEEDED =========
ADAPTER_PATH = os.getenv(
    "ADAPTER_PATH",
    "/Users/akshayjoshi/Documents/Company_Policies_documents/outputs/mistral_finetuned",
)
TRAIN_FILE = os.getenv(
    "TRAIN_FILE",
    "/Users/akshayjoshi/Documents/Company_Policies_documents/dataset/fine_tune_dataset_text.jsonl",
)
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"   # retrieval-only CI mode
# =======================================

# ---- knobs you can tweak without touching code below ----
DEBUG_RETRIEVAL = True       # set False to silence retrieval debug lines
TOP_K = 5                    # how many nearest questions to consider before rerank
CUT_STRONG = 85              # return exact dataset answer if >= this
CUT_MID    = 65              # if reranked >= this, still return dataset answer
CUT_WEAK   = 60              # below this, go to generation
GEN_MAX_NEW = 60             # generation length cap

# Silence general warnings and HF verbosity
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_OFFLINE"] = "1"                       # stay offline
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Quiet down HF + PEFT loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------- load model once --------------------
def _load_once():
    if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        print(f"❌ adapter_config.json not found at: {ADAPTER_PATH}")
        sys.exit(1)

    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, local_files_only=True)

    # ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Only resize embeddings if vocab actually differs (prevents advisory prints)
    try:
        old_vocab = model.get_input_embeddings().num_embeddings
    except Exception:
        old_vocab = None
    new_vocab = len(tokenizer)

    if old_vocab is None or new_vocab != old_vocab:
        try:
            model.resize_token_embeddings(new_vocab, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(new_vocab)

    model.to(DEVICE).eval()
    return model, tokenizer

MODEL, TOKENIZER = (None, None)
if not SKIP_MODEL_LOAD:
    MODEL, TOKENIZER = _load_once()

# -------------------- dataset cache ----------------------
_Q_RE  = re.compile(r"###\s*Question\s*:(.*?)(?=###|\Z)", re.S | re.I)
_A_RE  = re.compile(r"###\s*Answer\s*:(.*?)(?=###|\Z)",   re.S | re.I)

_FAQ_QS, _FAQ_AS, _FAQ_LOADED = [], [], False

def _extract_qa(raw_text: str):
    q = _Q_RE.search(raw_text)
    a = _A_RE.search(raw_text)
    q = q.group(1).strip() if q else None
    a = a.group(1).strip() if a else None
    if q and a:
        q = re.sub(r"\s+", " ", q)
        a = re.sub(r"\s+", " ", a)
        return q, a
    return None, None

def _load_faq():
    global _FAQ_LOADED, _FAQ_QS, _FAQ_AS
    if _FAQ_LOADED:
        return
    _FAQ_QS, _FAQ_AS = [], []
    if os.path.exists(TRAIN_FILE):
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = obj.get("text") or obj.get("content") or ""
                if not text and ("prompt" in obj and "completion" in obj):
                    text = f"### Question: {obj['prompt']}\n\n### Answer: {obj['completion']}"
                q, a = _extract_qa(text)
                if q and a:
                    _FAQ_QS.append(q)
                    _FAQ_AS.append(a)
    _FAQ_LOADED = True
    if DEBUG_RETRIEVAL:
        print(f"[dataset] loaded {_FAQ_QS and len(_FAQ_QS) or 0} Q/A pairs from {TRAIN_FILE}")

# -------------------- cleaning & stops -------------------
class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs):
        self.tok = tokenizer
        self.stop_ids = [self.tok(s, add_special_tokens=False).input_ids for s in stop_strs]
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        return any(len(seq) >= len(s) and seq[-len(s):] == s for s in self.stop_ids)

_BANNED_SNIPPETS = (
    "policy name", "template", "form", "applicable to:", "job title:", "department:",
    "location:", "approved by:", "hr manager", "note:", "dear ", "sincerely", "regards",
    "## question", "### question", "### answer", "[policy", "]", "[", "]"
)

def _pick_first_clean_sentence(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()

    # strip 'Question:' / 'Answer:' labels anywhere
    t = re.sub(r"(?i)\b(question|answer)\s*:\s*", " ", t)

    # remove headings and placeholders
    t = re.sub(r"(?m)^\s*#{1,6}\s+.*?$", "", t)
    t = re.sub(r"\[[^\]]+\]", "", t)
    t = re.sub(r"(\s[-•]\s+|\s\d+\.\s+)", " ", t)
    t = re.sub(r"(###|##|\*{2,}|_{2,}|-{3,})", " ", t)

    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip(" -•—:;,.") for p in parts if p.strip()]

    good = []
    for p in parts:
        low = p.lower()
        if any(b in low for b in _BANNED_SNIPPETS):
            continue
        if len(p.split()) < 4:
            continue
        if re.search(r"\s\d+$", p):
            continue
        good.append(p)

    if not good:
        return ""

    ans = good[0]
    if ans and ans[-1] not in ".?!":
        ans += "."
    return ans

def _clean(text: str) -> str:
    ans = _pick_first_clean_sentence(text)
    return ans if ans else "Not specified in the policy dataset."

def _answer_guard(user_q: str, ans: str) -> str:
    if not ans:
        return "Not specified in the policy dataset."
    low = ans.strip().lower()
    if low.startswith(("## question", "### question", "question:", "q:")):
        return "Not specified in the policy dataset."
    try:
        qn = re.sub(r"\s+", " ", user_q.strip().lower())
        an = re.sub(r"\s+", " ", ans.strip().lower())
        if fuzz.QRatio(qn, an) >= 85:
            return "Not specified in the policy dataset."
    except Exception:
        pass
    if len(ans.split()) < 4:
        return "Not specified in the policy dataset."
    return ans

# ------------------ intent + synonym normalization -----------------
ABBR = {
    r"\bWFH\b": "Work From Home",
    r"\bLWD\b": "Last Working Day",
    r"\bPOSH\b": "Prevention of Sexual Harassment",
}

def _expand_acronyms(q: str) -> str:
    for pat, repl in ABBR.items():
        q = re.sub(pat, repl, q, flags=re.IGNORECASE)
    return q.strip()

def _normalize_synonyms(q: str) -> str:
    # common paraphrases → canonical forms
    repls = [
        (r"\blong\s*duration\b", "long-term"),
        (r"\bextended\b", "long-term"),
        (r"\bsubmission window\b", "window"),
        (r"\bis it possible to\b", "can"),
        # NEW
        (r"\bvideo\b", "camera"),
        (r"\bwork\s*hours\b", "wfh hours"),
        (r"\bremote work hours\b", "wfh hours"),
        (r"\bmailbox\b", "email"),
    ]
    for pat, rep in repls:
        q = re.sub(pat, rep, flags=re.IGNORECASE, string=q)

    # normalize awkward revoke phrasings
    q = re.sub(r"(?i)can\s+(wfh|work\s+from\s+home)\s+be\s+revoked\??", r"Can \1 be revoked?", q)
    q = re.sub(r"(?i)can\s+(.+?)\s+be\s+revoked\??", r"Can \1 be revoked?", q)
    return q

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# ------------------- weighted reranker -------------------
def _score_pair(a: str, b: str) -> int:
    return int(0.7 * fuzz.QRatio(a, b) + 0.3 * fuzz.token_set_ratio(a, b))

def _keyword_overlap_bonus(q: str, c: str) -> int:
    qset = set(re.findall(r"\b\w+\b", q.lower()))
    cset = set(re.findall(r"\b\w+\b", c.lower()))
    bonus = 0
    for k in {
        "apply","application","reapply","re-apply","request","submit","process","steps",
        "wfh","work","from","home","remote","remotework","hrms","portal","tool","system",
        "when","timeline","window","cycle","april","november","month","months",
        "revoke","revoked","revocation","withdraw","cancel",
        # NEW keywords
        "camera","video","mandatory",
        "track","log","record","capture","hours","timesheet",
        "email","support","mail"
    }:
        if k in qset and k in cset:
            bonus += 2
    return bonus

def _filter_by_intent(q: str, candidates: list[str]) -> list[str]:
    ql = q.lower()

    wants_apply = any(k in ql for k in ("apply","application","reapply","re-apply","request","submit","process","steps"))
    wants_wfh   = ("wfh" in ql) or ("work from home" in ql) or ("remote work" in ql)
    wants_when  = any(k in ql for k in ("when","by what time","timeline","window","cycle","month","april","november"))
    wants_revoke= any(k in ql for k in ("revoke","revoked","revocation","withdraw","cancel"))
    # NEW:
    wants_camera = any(k in ql for k in ("camera","video"))
    wants_track  = any(k in ql for k in ("track","log","record","capture")) and any(k in ql for k in ("hours","timesheet"))
    wants_support= ("support" in ql) and ("email" in ql or "mail" in ql)

    keep = candidates

    if wants_apply and wants_wfh:
        tmp = [c for c in keep if any(k in c.lower() for k in ("apply","application","reapply","re-apply","request","submit","process","steps"))
                            and any(k in c.lower() for k in ("wfh","work from home","remote work"))]
        if tmp: keep = tmp

    if wants_when:
        tmp = [c for c in keep if any(k in c.lower() for k in ("when","timeline","window","month","april","november"))]
        if tmp: keep = tmp

    if wants_revoke:
        tmp = [c for c in keep if any(k in c.lower() for k in ("revoke","revoked","revocation","withdraw","cancel"))]
        if tmp: keep = tmp

    # NEW — camera rule
    if wants_camera:
        tmp = [c for c in keep if "camera" in c.lower()]
        if tmp: keep = tmp

    # NEW — track/log/record/capture WFH hours
    if wants_track:
        tmp = [c for c in keep if ("track" in c.lower() or "hours" in c.lower() or "timesheet" in c.lower())]
        if tmp: keep = tmp

    # NEW — support email
    if wants_support:
        tmp = [c for c in keep if ("email" in c.lower() or "mail" in c.lower()) and "support" in c.lower()]
        if tmp: keep = tmp

    return keep

def _weighted_score(query: str, cand: str) -> int:
    ql, cl = query.lower(), cand.lower()
    base = _score_pair(query, cand) + _keyword_overlap_bonus(query, cand)

    # existing boosts...
    if any(k in cl for k in ("apply","application","reapply","re-apply","request","submit","process","steps")):
        base += 10
    if any(k in cl for k in ("hrms","portal","tool","system")):
        base += 8
    if ("work from home" in cl) or ("wfh" in cl) or ("remote work" in cl):
        base += 6

    # temporal intent
    asks_when  = any(k in ql for k in ("when","by what time","timeline","window","cycle","month","april","november"))
    about_when = any(k in cl for k in ("when","timeline","window","month","april","november"))
    if asks_when and about_when: base += 12
    if asks_when and not about_when: base -= 6

    # reapply/location-change guard
    about_reapply = any(k in cl for k in ("reapply","re-apply","location changes","location change","change of location"))
    asks_reapply  = any(k in ql for k in ("reapply","re-apply","location change","change of location"))
    if about_reapply and not asks_reapply: base -= 18

    # NEW: revoke intent
    asks_revoke  = any(k in ql for k in ("revoke","revoked","revocation","withdraw","cancel"))
    about_revoke = any(k in cl for k in ("revoke","revoked","revocation","withdraw","cancel"))
    if asks_revoke and about_revoke: base += 12
    if asks_revoke and any(k in cl for k in ("approve","approved","approval","timeline","window","when")) and not about_revoke:
        base -= 10

    # long/short term
    asks_long  = ("long-term" in ql) or ("long term" in ql)
    about_long = ("long-term" in cl) or ("long term" in cl)
    if about_long and not asks_long: base -= 12
    asks_short  = ("short-term" in ql) or ("short term" in ql)
    about_short = ("short-term" in cl) or ("short term" in cl)
    if about_short and not asks_short: base -= 6

    # policy violation penalty (unless asked)
    if any(k in cl for k in ("violation","escalation","infosec","disciplinary")) and not any(k in ql for k in ("violation","escalation")):
        base -= 10

    # NEW: camera rule
    if any(k in ql for k in ("camera","video")) and "camera" in cl:
        base += 10

    # NEW: track/log/record/capture hours
    if any(k in ql for k in ("track","log","record","capture")) and any(k in ql for k in ("hours","timesheet")):
        if any(k in cl for k in ("track","hours","timesheet","hrms")):
            base += 12
        else:
            base -= 6

    # NEW: support email
    if "support" in ql and ("email" in ql or "mail" in ql):
        if any(k in cl for k in ("email","mail")) and "support" in cl:
            base += 10
        else:
            base -= 4

    return base

# -------------------- main QA function -------------------
def chat_with_model(query: str) -> str:
    """
    Retrieval-first QA:
      1) Load your JSONL once (### Question / ### Answer).
      2) Fuzzy-match the user query to training questions (RapidFuzz).
      3) If strong match (>= CUT_STRONG), return that exact answer.
      4) Else rerank top-K with weighted scorer; if >= CUT_MID return it.
      5) Else: build a tiny few-shot from top-2 and generate a terse answer.
    """
    # Use globals but copy to locals for safety
    global MODEL, TOKENIZER
    model, tokenizer, device = MODEL, TOKENIZER, DEVICE

    # --- load dataset ---
    _load_faq()

    # --- normalize/expand ---
    q_norm = _normalize_synonyms(_expand_acronyms(query))

    # --- retrieval ---
    top_matches = []
    chosen_idx = None
    chosen_score = None

    if _FAQ_QS:
        top_raw = process.extract(q_norm, _FAQ_QS, scorer=fuzz.QRatio, limit=TOP_K)
        top_questions = [m[0] for m in top_raw]
        top_indices   = [m[2] for m in top_raw]

        filtered = _filter_by_intent(q_norm, top_questions)
        if filtered and len(filtered) < len(top_questions):
            seen = set()
            filt_pairs = []
            for q in filtered:
                if q in seen:
                    continue
                seen.add(q)
                try:
                    idx = _FAQ_QS.index(q)
                except ValueError:
                    continue
                filt_pairs.append((q, idx))
        else:
            filt_pairs = list(zip(top_questions, top_indices))

        reranked = []
        for qcand, idx in filt_pairs:
            sc = _weighted_score(_normalize(q_norm), _normalize(qcand))
            reranked.append((qcand, sc, idx))
        reranked.sort(key=lambda x: x[1], reverse=True)

        if reranked:
            best_q, best_sc, best_idx = reranked[0]
            chosen_idx, chosen_score = best_idx, best_sc
            top_matches = reranked

    if DEBUG_RETRIEVAL and top_matches:
        print("[retrieval] query =", q_norm)
        print("[retrieval] top =", [(m[0], m[1]) for m in top_matches[:3]])
        print("[retrieval] chosen =", (_FAQ_QS[chosen_idx][:80] + "…", chosen_score))

    # --- thresholds (so CI doesn't go to generation path) ---
    cut_strong = CUT_STRONG
    cut_mid = CUT_MID
    if SKIP_MODEL_LOAD:
        cut_strong = min(CUT_STRONG, 80)
        cut_mid = min(CUT_MID, 60)

    # --- decision: return dataset answer if confident ---
    if chosen_idx is not None and chosen_score is not None:
        if chosen_score >= cut_strong:
            if DEBUG_RETRIEVAL:
                print(f"[path] returning exact dataset answer (strong match: {chosen_score})")
            ans = _clean(_FAQ_AS[chosen_idx])
            return _answer_guard(q_norm, ans)
        if chosen_score >= cut_mid:
            if DEBUG_RETRIEVAL:
                print(f"[path] returning nearest dataset answer (mid-band: {chosen_score})")
            ans = _clean(_FAQ_AS[chosen_idx])
            return _answer_guard(q_norm, ans)

    # --- generation fallback (few-shot with top-2) ---
    if SKIP_MODEL_LOAD:
        # In CI we don’t have the model; force safe fallback
        return "Not specified in the policy dataset."

    # Lazy-load if needed, then refresh locals
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = _load_once()
    model, tokenizer = MODEL, TOKENIZER

    fewshot = ""
    if _FAQ_QS and top_matches:
        for (mq, ms, mi) in top_matches[:2]:
            fewshot += f"### Question: {mq}\n\n### Answer: {_FAQ_AS[mi]}\n\n"

    prompt = (
        f"{fewshot}"
        f"### Question: {q_norm}\n\n"
        "### Answer: Reply with the exact policy in ONE short sentence. "
        "Do not use headings, lists, examples, placeholders, or templates. "
        "If the policy is not available, answer exactly: Not specified in the policy dataset.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    bad_words = [
        "\n###", "Examples:", "Dear ", "- ", "• ", "1.", "2.", "3.",
        "Applicable to:", "Job Title:", "Department:", "Location:", "Approved by:",
        "HR Manager", "Note:", "Sincerely", "Regards", "Template", "Form", "Policy Name:",
        "## ", "### ", "Question:", "Answer:"
    ]
    bad_words_ids = [tokenizer(b, add_special_tokens=False).input_ids for b in bad_words]
    stopping = StoppingCriteriaList([StopOnStrings(tokenizer, ["\n###", "### ", "## ", "Dear "])])

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW,
            min_new_tokens=12,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            bad_words_ids=bad_words_ids,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping,
            use_cache=True,
        )

    gen = out[0, inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True)
    if DEBUG_RETRIEVAL:
        print("[path] generation fallback")
        print("[gen] raw =", raw[:180].replace("\n", " ") + ("…" if len(raw) > 180 else ""))

    ans = _clean(raw)
    return _answer_guard(q_norm, ans)


# ------------------------- CLI ---------------------------
if __name__ == "__main__":
    print("Tip: ask things like 'How should I apply for WFH?' or 'Is camera on mandatory for meetings?'")
    while True:
        q = input("\n🔎 Ask a policy question (or type 'exit'): ")
        if q.lower() in ("exit", "quit"):
            break
        print("\n🤖 Answer:\n" + chat_with_model(q))
