import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from fpdf import FPDF
import os

# If you are on Windows and Tesseract is not in your PATH, uncomment and edit the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_and_create_pdf(input_pdf_path, output_pdf_path, lang='eng'):
    """
    Extracts text from a PDF (including text inside images) using OCR 
    and saves it into a new PDF file.
    
    :param input_pdf_path: Path to the original PDF file.
    :param output_pdf_path: Path where the new PDF will be saved.
    :param lang: Language code for Tesseract (e.g., 'eng', 'fra', 'spa', 'chi_sim').
    """
    if not os.path.exists(input_pdf_path):
        print(f"Error: The file {input_pdf_path} does not exist.")
        return

    print(f"Opening PDF: {input_pdf_path}...")
    # Open the PDF using PyMuPDF
    doc = fitz.open(input_pdf_path)
    total_pages = len(doc)
    extracted_text = ""

    print(f"Starting OCR on {total_pages} pages. This may take a while...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # Render page to an image with 300 DPI for high-quality OCR
        # Higher DPI = better text recognition, but slower processing
        pix = page.get_pixmap(dpi=300) 
        
        # Convert PyMuPDF pixmap to a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Perform OCR using Tesseract
        print(f"Processing page {page_num + 1}/{total_pages}...")
        page_text = pytesseract.image_to_string(img, lang=lang)
        
        # Add page separator and text to our main string
        extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"

    doc.close()

    print("Generating new PDF with extracted text...")
    
    # Create a new PDF using fpdf2
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # fpdf2 standard fonts only support Latin-1 encoding. 
    # We encode/decode to replace any weird OCR artifacts or unsupported Unicode characters 
    # so the PDF generation doesn't crash.
    clean_text = extracted_text.encode('latin-1', 'replace').decode('latin-1')
    
    # multi_cell automatically wraps text to the next line when it hits the page margin
    pdf.multi_cell(0, 10, clean_text)
    
    # Save the new PDF
    pdf.output(output_pdf_path)
    print(f"Success! Extracted text saved to: {output_pdf_path}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Change these paths to your actual files
    INPUT_FILE = r"F:\Amin_Projects\University\NLP_Project\Real_Project\data\1. Ayin-nameh-ha va Shiveh-nameh-haaye yekparcheh (tamam doreh-haaye tahsili)\Ayeen-nameh ye yekpaarcheh ye tamaam-e doreh-haaye tahsili baraaye daaneshjooyaan-e varoodi 1402 va ba'd az aan.pdf"   
    
    OUTPUT_FILE = "extracted_text_output.pdf" 
    
    # If your document is in a language other than English, change 'eng' 
    # (e.g., 'spa' for Spanish, 'fra' for French). 
    # Note: You must install the language pack in Tesseract for other languages.
    LANGUAGE = 'eng' 
    
    extract_text_and_create_pdf(INPUT_FILE, OUTPUT_FILE, lang=LANGUAGE)