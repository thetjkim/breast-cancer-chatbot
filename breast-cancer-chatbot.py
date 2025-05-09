# breast-cancer-chatbot.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import hub
from pathlib import Path

# 1. Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables."



# 2. Load and index the PDF documents using FAISS vector store
def load_all_pdfs(directory="pdfs"):
    all_docs = []
    for file in Path(directory).glob("*.pdf"):
        loader = PyMuPDFLoader(str(file))
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
    
@st.cache_resource
def load_vector_db():
    # Load PDF documents
    documents = load_all_pdfs("pdfs") 

    # Split text into manageable chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Create embeddings using OpenAI's embedding model
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.from_documents(chunks, embedding)
    return db

# 3. Set up the LangChain RAG pipeline
@st.cache_resource
def setup_chain():
    # Load the document retriever (FAISS vector search)
    db = load_vector_db()
    retriever = db.as_retriever()

    # Load the RAG prompt template from LangChain Hub
    # prompt = hub.pull("rlm/rag-prompt")

    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template("""
    You are a highly knowledgeable research assistant with expertise in oncology, especially breast cancer.
    Use the context below to answer the question as thoroughly and informatively as possible.
    
    Even if the exact answer is not stated, make your best educated guess based on related context and your medical knowledge.
    Avoid saying "The context does not mention" unless absolutely necessary.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """)

    # Initialize the OpenAI LLM (ChatGPT-4)
    llm = ChatOpenAI(model="gpt-4", temperature=0.1)

    # Create the RetrievalQA chain combining LLM, retriever, and prompt
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# 4. Streamlit UI for the chatbot
st.title("🩺 Breast Cancer Research Chatbot")

# Initialize the QA chain
qa_chain = setup_chain()

# Input box for user's question
query = st.text_input("Ask a question about breast cancer research:")

# If user submits a query, process it using the QA chain
if query:
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": query})
        st.markdown("### 💡 Answer:")
        st.write(result["result"])
