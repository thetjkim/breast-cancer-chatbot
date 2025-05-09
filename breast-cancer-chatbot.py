# streamlit_chatbot.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain import hub

# 1. 환경 설정
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."

# 2. 벡터 DB 로드 또는 생성
@st.cache_resource

def load_vector_db():
    vector_dir = "./chroma_breast"
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(vector_dir):
        return Chroma(
            collection_name="breast-cancer",
            persist_directory=vector_dir,
            embedding_function=embedding
        )
    else:
        loader = PyMuPDFLoader("nih_breast_cancer.pdf")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        return Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name="breast-cancer",
            persist_directory=vector_dir
        )

# 3. LangChain 구성
@st.cache_resource

def setup_chain():
    db = load_vector_db()
    retriever = db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# 4. Streamlit 인터페이스
st.title("🩺 Breast Cancer Research Chatbot")

qa_chain = setup_chain()

query = st.text_input("Ask a question about breast cancer research:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.markdown("### 💡 Answer:")
        st.write(result["result"])
