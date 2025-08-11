from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS 
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


chunk_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/cleaned_text")
index_path = "internal_document_chatbot/faiss_index_recursive"


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
all_chunks = []
metadata_list = []


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
    

for file in chunk_dir.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) > 30:
            all_chunks.append(Document(
                page_content=chunk.strip(),
                metadata={
                    "source_file": file.name,
                    "chunk_id": i,
                    "chunk_preview": chunk.strip()[:100],
                    "tag": assign_tag(chunk)
                }
            ))


# --- Manual edge case chunks to boost retrieval ---
manual_chunks = [
    (
        "Employees are required to mark all types of leaves — including Sick Leave, Earned Leave, and Work From Home — using the HRMS portal. Leave must be applied at least one day in advance, except in emergency cases. Failure to mark leave in HRMS will be treated as unapproved absence. For WFH requests, select the 'Remote Request' tab within Attendance, specify the date range, and submit for manager auto-approval.",
        "leaves_hrms"
    ),
    (
        "Engaging in dual employment is strictly prohibited. If any employee is found working with another organization during active employment (including while on WFH or leave), it will be treated as gross misconduct. Penalties include immediate suspension of remote access, HRBP inquiry, and possible termination following internal investigation.",
        "wfh_dual_employment"
    ),
    (
        "Employees who change their home location (city or state) must reapply for Work From Home approval. This includes submission of updated address proof, network readiness checklist, and a revised compliance declaration form. Approval is contingent upon business feasibility and reviewed by the BU Head.",
        "wfh_location_change"
    ),
    (
        "Employees working remotely must use a secure and stable internet connection with minimum 50 Mbps speed. Use of mobile hotspots is discouraged. IT will perform periodic speed checks and device audits. Personal Wi-Fi routers must be password-protected. Any breach of data due to poor network hygiene will be escalated to InfoSec.",
        "wfh_network_policy"
    ),
    (
        "Only confirmed employees with at least 3 months of service are eligible for long-term WFH. Probationary employees or those undergoing Performance Improvement Plan (PIP) are not eligible for remote work unless explicitly approved by the HRBP and Reporting Manager. Exceptions are rare and must be justified in writing.",
        "wfh_eligibility"
    ),
    (
        "WFH employees must log into the corporate VPN before accessing any internal systems. Additionally, endpoint monitoring tools like ES Plus must be active throughout working hours. Logs are audited weekly. Unusual login patterns or usage gaps beyond 45 minutes will trigger automated alerts to IT and HR.",
        "wfh_security"
    ),
    (
        "Emergency WFH can be requested via direct email to the reporting manager and HRBP when HRMS access is unavailable. Approval must be documented post-facto in the HRMS system within 2 working days. Abuse of emergency WFH will result in policy violation warnings.",
        "wfh_short_term"
    ),
    (
        "If an employee on WFH fails to meet response SLAs for more than 2 days or is found unresponsive during core hours, their remote access may be revoked. HRBP will issue a warning, followed by a mandatory one-week WFO period before re-evaluation.",
        "wfh_escalation"
    ),
        (
        "Employees are strictly prohibited from using personal laptops for official work. All work-related activities must be conducted using company-issued and monitored devices only. Any exceptions must be approved in writing by the IT Head and the HRBP. Violations may lead to revocation of access privileges and formal disciplinary action.",
        "device_policy"
    ),
    (
        "Attendance records — including log-in/log-out times, leave applications, and WFH status — are audited monthly by the HR Analytics team. Any discrepancies are escalated to HRBPs and relevant managers. Employees are advised to regularly verify their attendance status in the HRMS portal.",
        "attendance_audit"
    ),
    (
        "In the event of an emergency WFH request made outside the HRMS system, employees must email both their reporting manager and HRBP with justification. A post-facto request must be submitted in HRMS within 2 working days along with proof of the emergency. Repeated misuse may lead to rejection of future WFH privileges.",
        "wfh_short_term"
    ),
    (
        "Remote employees must maintain a minimum internet speed of 50 Mbps. Use of unstable mobile hotspots is not recommended. IT may perform periodic bandwidth checks. Failure to maintain required speed may result in interrupted access to VPN and systems.",
        "wfh_network_policy"
    )
]

# Append manually crafted documents to vector store
start_idx = len(all_chunks)  # avoid ID conflicts
for i, (text, tag) in enumerate(manual_chunks):
    all_chunks.append(Document(
        page_content=text,
        metadata={
            "source_file": "manual_chunk.txt",
            "chunk_id": start_idx + i,
            "chunk_preview": text[:100],
            "tag": assign_tag(chunk)
        }
    ))

            
##Embedding
embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_chunks, embedding_model)

##Save FAISS Index
vectorstore.save_local(index_path)
print(f"Recursive chunked index saved at: {index_path} with {len(all_chunks)} chunks.")