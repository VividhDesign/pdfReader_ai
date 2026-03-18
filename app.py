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

# API Keys load karna (Local ke liye .env, Cloud ke liye Secrets)
load_dotenv()

# Page Configuration
st.set_page_config(page_title="pdfReader_ai", page_icon="🤖", layout="wide")

# --- PROFESSIONAL UI CLEANUP (Hides Deploy/Made with Streamlit) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Main Title
st.title("🤖 pdfReader_ai")
st.markdown("---")

# Sidebar for Uploading PDF & Branding
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    process_button = st.button("Analyze PDF", use_container_width=True)
    
    # --- CREATOR SECTION ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Vividh Yadav**")
    st.caption("AI & ML | BIT Mesra")
    
    with st.expander("📬 Contact Details"):
        st.write("📧 [vividh50@gmail.com](mailto:vividh50@gmail.com)")
        st.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/vividh-yadav-866a44380/)")
        st.markdown("[💻 GitHub Portfolio](https://github.com/VividhDesign)")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Caching Models for Performance
@st.cache_resource
def get_tools():
    # Fetching API Keys from environment (works for both local and streamlit cloud secrets)
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.1,
        groq_api_key=groq_api_key
    )
    return embeddings, llm

# Backend RAG Chain Logic
def get_rag_chain(pdf_path):
    embeddings, llm = get_tools()
    
    # Load and Split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Store (Ephemeral for the session)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # Define Prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a professional assistant. Answer the question based ONLY on the provided context.\n\n"
        "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain Construction
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# File Processing Execution
if uploaded_file and process_button:
    with st.spinner("Analyzing document... Hang tight!"):
        # Temporary storage
        temp_file = "temp_pdf_storage.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Build the RAG chain and store in session
        st.session_state.rag_chain = get_rag_chain(temp_file)
        st.success("PDF Processed Successfully!")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
if prompt_input := st.chat_input("Ask a question about the document..."):
    if "rag_chain" not in st.session_state:
        st.warning("Please upload and process a PDF to start chatting.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Generate and Append AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt_input)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
