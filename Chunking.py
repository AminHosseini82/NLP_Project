from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # حدود 300 تا 500 کلمه/توکن فارسی
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_text(cleaned_text)
# در نهایت این چانک‌ها را به شکل دیکشنری همراه با متادیتا ذخیره کن
documents = [{"text": chunk, "metadata": {"title": "آیین نامه...", "page": page_num}} for chunk in chunks]