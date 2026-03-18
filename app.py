import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# API Keys loading
load_dotenv()

# Page Configuration
st.set_page_config(page_title="pdfReader_ai", page_icon="🤖", layout="wide")

# --- UI CLEANUP (Professional Look) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Mobile adjustment */
            .stChatInputContainer {padding-bottom: 20px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Main Header
st.title("🤖 pdfReader_ai")
st.caption("Multi-PDF Contextual Chat | Llama 3.3 & Groq")

# --- INITIALIZE SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Caching Models for Performance
@st.cache_resource
def get_tools():
    groq_api_key = os.getenv("GROQ_API_KEY")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    return embeddings, llm

# Logic to Add PDF and Handle Empty/Image PDFs
def add_pdf_to_knowledge(pdf_path):
    embeddings, llm = get_tools()
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            st.error("❌ No text found in this PDF. It might be a scanned image.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if not splits:
            st.warning("⚠️ Could not extract meaningful text chunks from this document.")
            return

        # Vectorstore Persistence Logic
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
        else:
            st.session_state.vectorstore.add_documents(
                documents=splits,
                embedding=embeddings
            )
        
        # Update the RAG Chain with new context
        retriever = st.session_state.vectorstore.as_retriever()
        prompt = ChatPromptTemplate.from_template(
            "Answer based on ALL provided documents. Context: {context}\nQuestion: {input}\nAnswer:"
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        st.session_state.rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        st.success("✅ Document processed and added to context!")

    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")

# --- MAIN PAGE UPLOAD (Best for Mobile) ---
st.markdown("### 📄 Step 1: Upload & Process")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
process_button = st.button("🚀 Add to Chat Context", use_container_width=True)

if uploaded_file and process_button:
    with st.spinner(f"Reading {uploaded_file.name}..."):
        temp_file = "temp_pdf_storage.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        add_pdf_to_knowledge(temp_file)

st.divider()

# --- SIDEBAR (Branding & Memory Control) ---
with st.sidebar:
    st.header("⚙️ Memory Control")
    if st.button("🗑️ Clear All Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        if "rag_chain" in st.session_state:
            del st.session_state.rag_chain
        st.rerun()

    st.divider()
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Vividh Yadav**")
    st.caption("B.Tech AI & ML | BIT Mesra")
    
    with st.expander("📬 Contact Me"):
        st.write("📧 vividh50@gmail.com")
        st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/vividh-yadav-866a44380/)")
        st.markdown("[💻 GitHub](https://github.com/VividhDesign)")

# --- CHAT INTERFACE ---
st.markdown("### 💬 Step 2: Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask a question about your documents..."):
    if "rag_chain" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
