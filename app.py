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

# --- UI CLEANUP ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stChatInputContainer {padding-bottom: 20px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🤖 pdfReader_ai")
st.caption("Multi-PDF Contextual Chat | Llama 3.3 & Groq")

# --- INITIALIZE SESSION STATES ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

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

def add_pdf_to_knowledge(pdf_path):
    embeddings, llm = get_tools()
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            st.error("❌ No text found in this PDF. It might be a scanned image.")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        if not splits:
            st.warning("⚠️ Could not extract meaningful text chunks.")
            return False

        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
        else:
            st.session_state.vectorstore.add_documents(
                documents=splits
            )
        
        # 🔥 FIX 2: Added MMR search for diverse retrieval across multiple PDFs & increased top K
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )
        
        # 🔥 FIX 3: Prompt updated to properly differentiate between documents
        prompt = ChatPromptTemplate.from_template(
            "You are an intelligent assistant. Use the following document extracts to answer the user's question.\n"
            "The context includes the source PDF filename for reference. Pay attention to all provided sources.\n\n"
            "CONTEXT:\n{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )
        
        def format_docs(docs):
            # Formats chunks with their actual source filename so the LLM doesn't get confused
            return "\n\n".join(f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}" for doc in docs)

        st.session_state.rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return True

    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        return False

# --- SIDEBAR (Memory Control Only) ---
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
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask a question or attach a PDF...", 
    accept_file="multiple", 
    file_type=["pdf"]
)

if user_input:
    # 1. Agar user ne file upload ki hai
    if user_input.files:
        with st.chat_message("user"):
            file_msg = f"📎 *Uploaded {len(user_input.files)} document(s)*"
            st.markdown(file_msg)
            st.session_state.messages.append({"role": "user", "content": file_msg})
            
        with st.spinner("Processing documents..."):
            for file in user_input.files:
                # 🔥 FIX 1: Unique temp file name so metadata doesn't clash
                temp_file = f"temp_{file.name}"
                with open(temp_file, "wb") as f:
                    f.write(file.getvalue())
                
                success = add_pdf_to_knowledge(temp_file)
                if success:
                    st.toast(f"✅ Processed {file.name}")
                
                # Cleanup temporary file after processing to save space
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
    # 2. Agar user ne koi text bheja hai
    if user_input.text:
        if "rag_chain" not in st.session_state:
            st.warning("Please attach a PDF using the '+' icon first! 👈")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input.text})
            with st.chat_message("user"):
                st.markdown(user_input.text)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke(user_input.text)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
