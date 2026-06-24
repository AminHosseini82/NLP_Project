import os
import fitz
from hazm import Normalizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import DATA_DIR, DB_DIR, EMBEDDING_MODEL

normalizer = Normalizer()

def process_and_store_documents():
    documents = []
    
    # 1. خواندن تمام فایل‌های PDF از پوشه data
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                text = doc.load_page(page_num).get_text()
                if not text.strip():
                    continue # رد کردن صفحات خالی
                
                # 2. پاک‌سازی و نرمال‌سازی
                cleaned_text = normalizer.normalize(text)
                
                # 3. تقسیم به Chunk های مناسب (حدود 300 تا 500 توکن)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500, # حدود 400 کلمه فارسی
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = text_splitter.split_text(cleaned_text)
                
                # 4. افزودن متادیتا
                for chunk in chunks:
                    metadata = {
                        "source": filename,
                        "page": page_num + 1,
                        # استخراج ماده و تاریخ نیازمند پردازش الگو (Regex) روی متن است
                    }
                    documents.append({"text": chunk, "metadata": metadata})

    print(f"Total chunks created: {len(documents)}")

    # 5. تولید Embedding و ذخیره در دیتابیس برداری
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    print("Creating and saving Vector Database (Chroma)...")
    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=DB_DIR
    )
    print("Database created successfully!")

if __name__ == "__main__":
    process_and_store_documents()