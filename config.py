import os
from dotenv import load_dotenv


# مسیر پوشه داده‌ها و دیتابیس برداری
DATA_DIR = "data"
DB_DIR = "chroma_db"

# مدل Embedding برای تبدیل متن به بردار (پشتیبانی از فارسی)
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# کلید API برای Groq (رایگان)
# می‌توانی از سایت console.groq.com یک کلید رایگان بگیری
# GROQ_API_KEY = "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" 

# This loads the variables from .env into the environment
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

