import re
import PyPDF2

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, size=300, overlap=40):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def process_uploaded_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        chunks = []
        for page in reader.pages:
            if page.extract_text():
                chunks.extend(chunk_text(clean_text(page.extract_text())))
        return chunks
    else:
        return chunk_text(clean_text(file.read().decode("utf-8")))
