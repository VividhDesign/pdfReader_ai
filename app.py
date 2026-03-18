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

# API Keys load karna
load_dotenv()

# Page Configuration (Mobile-friendly layout)
st.set_page_config(page_title="pdfReader_ai", page_icon="🤖", layout="centered")

# --- UI CLEANUP ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* Mobile adjustment for chat input */
            .stChatInputContainer {padding-bottom: 20px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Main Title
st.title("🤖 pdfReader_ai")
st.caption("Chat with your documents in seconds using Llama 3.3 & Groq")
st.markdown("---")

# --- MAIN PAGE UPLOAD (Better for Mobile) ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
process_button = st.button("🚀 Process & Analyze", use_container_width=True)

st.markdown("---")

# Sidebar for Branding only
with st.sidebar:
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Vividh Yadav**")
    st.caption("B.Tech AI & ML | BIT Mesra")
    
    st.divider()
    with st.expander("📫 Contact & Links", expanded=True):
        st.write("📧 [vividh50@gmail.com](mailto:vividh50@gmail.com)")
        st.markdown("[🔗 LinkedIn Profile](https://www.linkedin.com/in/vividh-yadav-866a44380/)")
        st.markdown("[💻 GitHub Portfolio](https://github.com/VividhDesign)")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Caching Models
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

# RAG Logic
def get_rag_chain(pdf_path):
    embeddings, llm = get_tools()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based ONLY on the context.\n\nContext: {context}\n\nQuestion: {input}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# Processing
if uploaded_file and process_button:
    with st.spinner("Analyzing PDF..."):
        temp_file = "temp_pdf_storage.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.rag_chain = get_rag_chain(temp_file)
        st.success("Ready! Ask your questions below.")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask something about the PDF..."):
    if "rag_chain" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke(prompt_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
