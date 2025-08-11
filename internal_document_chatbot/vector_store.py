from pathlib import Path
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

# Paths
chunk_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/cleaned_text")
index_path = "internal_document_chatbot/faiss_index"

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -- Updated Tagging Function --
def assign_tag(chunk: str) -> str:
    chunk_lower = chunk.lower()

    if "remote request" in chunk_lower or "auto-approved" in chunk_lower or "short-term" in chunk_lower:
        return "wfh_short_term"
    elif "application window" in chunk_lower or "april 1–15" in chunk_lower or "november 1–15" in chunk_lower:
        return "wfh_long_term"
    elif "post approval" in chunk_lower or "after wfh is approved" in chunk_lower or "approved, you will receive":
        return "wfh_post_approval"
    elif "escalation" in chunk_lower or "revocation" in chunk_lower or "disciplinary" in chunk_lower:
        return "wfh_escalation"
    elif any(kw in chunk_lower for kw in ["vpn", "es plus", "endpoint", "monitoring", "hubstaff", "teramind"]):
        return "wfh_security"
    elif "camera" in chunk_lower or "video conferencing" in chunk_lower:
        return "wfh_camera"
    elif "@company.com" in chunk_lower or "support email" in chunk_lower:
        return "wfh_contact"
    elif "new employee" in chunk_lower or "confirmation status" in chunk_lower:
        return "wfh_eligibility"
    elif "maternity leave" in chunk_lower:
        return "maternity"
    else:
        return "general"


# Load & chunk documents
documents, metadata = [], []

for file in chunk_dir.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    for i, chunk in enumerate(text.split("\n\n")):
        chunk = chunk.strip()
        if len(chunk) > 30:
            tag = assign_tag(chunk)
            documents.append(chunk)
            metadata.append({
                "source_file": file.name,
                "chunk_id": i,
                "chunk_preview": chunk[:100],
                "tag": tag
            })

# Generate embeddings
print("Generating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
lc_documents = [Document(page_content=d, metadata=m) for d, m in zip(documents, metadata)]
vectorstore = FAISS.from_documents(lc_documents, embedding_model)

# Save index
vectorstore.save_local(index_path)
print(f"✅ Indexed {len(lc_documents)} documents with improved tagging to {index_path}")


