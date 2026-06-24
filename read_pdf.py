import fitz  # PyMuPDF
from hazm import Normalizer

normalizer = Normalizer()

def extract_and_clean_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    for page_num in range(len(doc)):
        text = doc.load_page(page_num).get_text()
        cleaned_text = normalizer.normalize(text)
        pages_data.append({"page_num": page_num + 1, "text": cleaned_text})
    return pages_data