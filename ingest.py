import os
import pytesseract
from pdf2image import convert_from_path
from hazm import Normalizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DATA_DIR, DB_DIR, EMBEDDING_MODEL

# مسیر فایل اجرایی Tesseract در ویندوز
# اگر Tesseract را در درایو دیگری نصب کرده‌ای، این مسیر را تغییر بده
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

normalizer = Normalizer()

def process_and_store_documents():
    documents = []
    
    print("⏳ در حال استخراج متن از فایل‌های PDF با استفاده از OCR... (این مرحله ممکن است زمان‌بر باشد)")
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            print(f"\n📄 پردازش فایل: {filename}")
            
            # تبدیل PDF به لیست تصاویر (رزولوشن 200 برای سرعت و دقت مناسب است)
            # نکته: اگر اینجا ارور داد، یعنی Poppler روی ویندوزت نصب یا در PATH نیست
            pages = convert_from_path(file_path, dpi=200 , poppler_path=r"C:\poppler-26.02.0\Library\bin")
            
            for page_num, image in enumerate(pages):
                # استخراج متن از تصویر با زبان فارسی (fas)
                text = pytesseract.image_to_string(image, lang='fas')
                
                if not text.strip():
                    continue # رد کردن صفحاتی که متنی ندارند
                    
                # چاپ 50 کاراکتر اول برای اطمینان از اینکه متن به هم ریخته (Gibberish) نیست
                print(f"--- نمونه متن صفحه {page_num + 1} ---")
                print(text[:50].replace('\n', ' '))
                
                # پاک‌سازی و نرمال‌سازی متن با hazm
                cleaned_text = normalizer.normalize(text)
                
                # تقسیم به Chunk های مناسب
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = text_splitter.split_text(cleaned_text)
                
                # افزودن متادیتا
                for chunk in chunks:
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                    }
                    documents.append({"text": chunk, "metadata": metadata})

    # جلوگیری از کرش کردن در صورتی که هیچ متنی پیدا نشد
    if not documents:
        print("\n❌ خطا: هیچ متنی استخراج نشد! بررسی کنید که Tesseract زبان فارسی را داشته باشد.")
        return

    print(f"\n✅ مجموع قطعات (Chunks) ساخته شده: {len(documents)}")

    # تولید Embedding و ذخیره در دیتابیس برداری
    print("⏳ در حال تولید بردارهای معنایی و ذخیره در ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"} # استفاده از فاصله کسینوسی که برای فارسی بهتر است
    )
    print("🎉 دیتابیس برداری با موفقیت ساخته شد!")

if __name__ == "__main__":
    process_and_store_documents()