from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
vector_db = Chroma.from_texts(
    texts=[doc["text"] for doc in documents], 
    embedding=embeddings, 
    metadatas=[doc["metadata"] for doc in documents]
)

# تابع جستجو که k سند برتر را برمی‌گرداند
retrieved_docs = vector_db.similarity_search_with_score(user_query, k=3)