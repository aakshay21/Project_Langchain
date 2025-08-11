import re
import json
from pathlib import Path

input_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/cleaned_text")
output_dir = Path("/Users/akshayjoshi/Documents/Company_Policies_documents/chunks_json")
output_dir.mkdir(parents=True, exist_ok=True)


def chunk_by_section(text: str, source_name: str) -> list[dict]:
    pattern = r"(?=\n?\d+\.\s+[A-Z])"
    sections = re.split(pattern, text)
    chunks = []

    for section in sections:
        section = section.strip()
        if len(section) > 30:
            title_match = re.match(r"^\d+\.\s+(.+)", section)
            title = title_match.group(1).strip() if title_match else "Untitled"
            chunks.append({
                "title": title,
                "content": section,
                "source": source_name
            })
    return chunks


def chunk_all_documents():
    for file in input_dir.glob("*.txt"):
        print(f"📄 Chunking: {file.name}")
        text = file.read_text(encoding="utf-8")
        chunks = chunk_by_section(text, file.name)

        # Save chunks to JSON
        output_path = output_dir / f"{file.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(chunks)} chunks to: {output_path}")


if __name__ == "__main__":
    chunk_all_documents()
