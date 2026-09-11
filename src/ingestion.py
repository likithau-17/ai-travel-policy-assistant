from pathlib import Path
import re


POLICY_DIR = Path("data/company_policy")


def get_metadata(source):
    if "india" in source.lower():
        country = "India"
    elif "us" in source.lower():
        country = "US"
    else:
        country = "Global"

    if "travel_policy" in source.lower():
        policy_type = "travel"
    elif "airport" in source.lower():
        policy_type = "airport"
    elif "approval" in source.lower():
        policy_type = "approval"
    elif "cancellation" in source.lower():
        policy_type = "cancellation"
    elif "employee" in source.lower():
        policy_type = "employee"
    elif "expense" in source.lower():
        policy_type = "expense"
    else:
        policy_type = "unknown"

    return {
        "country": country,
        "policy_type": policy_type,
    }


def load_policy_documents():
    documents = []

    for file_path in POLICY_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


def split_into_sections(text):
    pattern = r"(?m)^(?P<number>\d+)\.\s+(?P<title>.+)$"

    matches = list(re.finditer(pattern, text))

    sections = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        section_text = text[start:end].strip()

        sections.append(
            {
                "section_number": match.group("number"),
                "section_title": match.group("title").strip(),
                "text": section_text,
            }
        )

    return sections

def build_chunks():
    chunks = []

    for document in load_policy_documents():
        metadata = get_metadata(document["source"])
        document_sections = split_into_sections(document["text"])

        for section in document_sections:
            chunks.append(
                {
                    "text": section["text"],
                    "metadata": {
                        "source": document["source"],
                        "country": metadata["country"],
                        "policy_type": metadata["policy_type"],
                        "section_number": section["section_number"],
                        "section_title": section["section_title"],
                    },
                }
            )

    return chunks

if __name__ == "__main__":
    chunks = build_chunks()

    print(f"Total chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n---")
        print("Metadata:")
        print(chunk["metadata"])
        print("Text:")
        print(chunk["text"])