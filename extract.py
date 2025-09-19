import re
import fitz
from pathlib import Path
from config import INPUT_DIR, CLEAN_DIR
from cleaner import pretext
import pandas as pd


def extract(file_path: str):
    with fitz.open(str(file_path)) as doc:
        txt = ""
        for page_num, page in enumerate(doc, 1):
            page_txt = page.get_text("text")
            if page_txt:
                txt += page_txt + "\n"
            else:
                print(f"No text found in {page_num} of {file_path}")

    return txt.strip()


# def extract_after_requirement(text):
#     """
#     Extracts everything after 'requirement' (case-insensitive).
#     Handles both str and list inputs.
#     """
#     if isinstance(text, list):
#         text = " ".join(text)

#     text_lower = text.lower()
#     match = re.search(r"requirement[s]?\s*[:\-]?\s*(.*)", text_lower, re.IGNORECASE | re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return text.strip()


def process_pdf_to_csv():
    input_dir = Path(INPUT_DIR)
    clean_dir = Path(CLEAN_DIR)
    clean_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for pdf_file in input_dir.glob("*.pdf"):
        text = extract(pdf_file)
        clean = pretext(text)
        if isinstance(clean, list):
            clean = " ".join(clean)
                
        # desc_after_req = extract_after_requirement(clean)  # ✅ new logic
        data.append({
            "filename": pdf_file.name,
            "description": clean
        })

    df = pd.DataFrame(data)
    output = clean_dir / "job_desc.csv"
    df.to_csv(output, index=False, encoding="utf-8")

    print(f"Extracted {len(data)} files and saved to {output}")


if __name__ == "__main__":
    process_pdf_to_csv()
