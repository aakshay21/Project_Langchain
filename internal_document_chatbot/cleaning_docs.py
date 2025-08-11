import os
from pathlib import Path
import docx
import fitz
import re


##Define input/output directories

pdf_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/Pdfs")
word_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/Docx")
output_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/cleaned_text")
output_dir.mkdir(parents=True,exist_ok = True)

##Text Cleaning

import re

def clean_text(text: str) -> str:
    # Initial whitespace cleanup
    text = text.replace("\xa0", " ")

    # Fix hyphenated line breaks (e.g., "On\n- Call" → "On-Call")
    text = re.sub(r"\n-\s*", "-", text)

    # Remove stray newlines (optional for flat structure)
    text = re.sub(r"\n", " ", text)

    # Normalize bullet symbols
    text = re.sub(r"[•●▪▶→➤➔]", "-", text)      # Convert all bullets to "-"
    text = re.sub(r"\s*--\s*", "\n- ", text)     # Convert double dashes to clean bullet
    text = re.sub(r"\s*-\s*-", "\n- ", text)     # Convert space-separated dashes to bullet

    # Ensure dash-bullets start on a new line (preserve in lists)
    text = re.sub(r"(?<!\w)-(?![\w\d])", "\n- ", text)

    # Remove noisy special characters but keep structure
    text = re.sub(r"[*_!#%^=<>|~`]", "", text)


    # Remove common footer lines
    text = re.sub(r"©.*?(Pvt|Ltd|LLP|Technologies).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(Confidential Document|Proprietary)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(www\.|http).*", "", text)
    text = re.sub(r"contact:.*", "", text, flags=re.IGNORECASE)

    # Remove generic email footer lines (but keep real POC emails)
    lines = text.splitlines()
    signature_triggers = ["Regards", "Thanks", "Best,", "Sincerely"]
    filtered = []
    for i, line in enumerate(lines):
        if any(trigger in line.strip() for trigger in signature_triggers):
            break  # skip everything after this line
        if re.match(r"(?i)^email:\s*(info|noreply)@.*$", line.strip()):
            continue
        filtered.append(line)
    text = "\n".join(filtered)

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()

    return text



##PDF Extraction

def extract_text_from_pdf(file_path: Path) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)


##Word Docs Extraction

def extract_text_from_docx(file_path: Path) -> str:
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return clean_text("\n".join(full_text))


##Save text to file

def save_text(filename: str, content: str):
    output_path = output_dir / f"{filename}.txt"
    with open(output_path,"w", encoding="utf-8") as f:
        f.write(content)


##Main Execution

def extract_all_documents():
    combined = []

    print("✅ PDF files found:", list(pdf_dir.glob("*.pdf")))
    print("✅ DOCX files found:", list(word_dir.glob("*.docx")))

    for file in pdf_dir.glob("*.pdf"):
        if file.suffix.lower() == ".pdf":
            print(f"Extracting PDF: {file.name}")
            text = extract_text_from_pdf(file)
            save_text(file.stem, text)
            print(f"✅ Added: {file.name} | Length: {len(text)} chars")
            combined.append(text)
    
    for file in word_dir.glob("*.docx"):
        if file.suffix.lower() == ".docx":
            print(f"Extracting DOCX: {file.name}")
            text = extract_text_from_docx(file)
            save_text(file.stem, text)
            print(f"✅ Added: {file.name} | Length: {len(text)} chars")
            combined.append(text)
    
    with open(output_dir / "all_docx_combined.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined))

    print("Extractaction complete. Cleaned files saved to:", output_dir)

print("✅ PDF files found:", list(pdf_dir.glob("*.pdf")))
print("✅ DOCX files found:", list(word_dir.glob("*.docx")))




if __name__ == "__main__":
    extract_all_documents()