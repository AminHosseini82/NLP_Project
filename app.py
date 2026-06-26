import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import ChatPromptTemplate
from config import DB_DIR, EMBEDDING_MODEL, GROQ_API_KEY

# بارگذاری دیتابیس و مدل‌ها (Cache برای سرعت بیشتر)
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    # llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
    return db, llm

db, llm = load_rag_components()

st.set_page_config(page_title="سامانه پرسش‌وپاسخ دانشگاه", page_icon="🎓", layout="centered")

st.markdown("""
<style>
    /* راست‌چین کردن کل بدنه سایت و تنظیم فونت */
    body, .stApp {
        direction: rtl;
        text-align: right;
        font-family: 'Tahoma', 'Vazir', 'B Nazanin', sans-serif;
    }
    
    /* راست‌چین کردن لیبل‌های ورودی متن (مانند 'سؤال شما:') */
    .stTextInput > div > div > div > div > p {
        text-align: right !important;
        direction: rtl;
    }
    
    /* راست‌چین کردن باک‌های پیام (مانند st.info و st.error) */
    .stAlert > div {
        direction: rtl;
        text-align: right;
    }

    /* تنظیم دکمه‌ها برای ظاهر بهتر در حالت فارسی */
    .stButton > button {
        float: right;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 سامانه پرسش‌وپاسخ هوشمند دانشگاه باهنر")
st.write("سوال خود را درباره آیین‌نامه‌ها و دستورالعمل‌ها بپرسید.")

query = st.text_input("سؤال شما:")

if st.button("جستجو") and query:
    with st.spinner("در حال جستجو و تولید پاسخ..."):
        # جستجوی اسناد مرتبط (K=6) به همراه محاسبه امتیاز شباهت
        search_query = f"query: {query}"
        
        results = db.similarity_search_with_score(query, k=6)
        
        if not results:
            st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
        else:
            context = ""
            sources = []
            # استخراج متن و متادیتا از نتایج بازیابی شده
            for doc, score in results:
                context += f"\nمتن: {doc.page_content}\n"
                # در ChromaDB فاصله (Distance) برمی‌گردد، مقدار کمتر یعنی شباهت بیشتر
                sources.append({
                    "source": doc.metadata.get('source', 'نامشخص'), 
                    "page": doc.metadata.get('page', 'نامشخص'), 
                    "distance_score": round(score, 4)
                })

            # تنظیم پرامپت دقیقاً طبق خواسته پروژه
            prompt_template = """
            تو یک دستیار هوشمند و دقیق برای پاسخگویی به سوالات دانشگاهی هستی.
            با دقت اطلاعات زیر را بخوان و بر اساس آن‌ها یک پاسخ کامل و روان به سوال کاربر بده.
            اگر پاسخ در متن زیر وجود دارد اما با کلمات دیگری بیان شده، مفهوم آن را استخراج کن.
            فقط اگر هیچ ارتباطی بین اطلاعات زیر و سوال کاربر وجود نداشت بگو: "پاسخ این سؤال در اسناد موجود یافت نشد."
            
            اطلاعات استخراج شده:
            {context}
            
            سوال کاربر: {query}
            """
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | llm
            
            response = chain.invoke({"context": context, "query": query})
            
            st.markdown("### پاسخ سیستم:")
            st.info(response.content)
            
            st.markdown("### 📚 منابع استفاده شده:")
            for s in sources:
                st.write(f"- **سند:** {s['source']} | **صفحه:** {s['page']} | **امتیاز شباهت (فاصله):** {s['distance_score']}")