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

# Page Configuration (Browser tab par kya dikhega)
st.set_page_config(page_title="Bobo's RAG Bot", page_icon="🤖")
st.title("🤖 Chat with Your PDF (RAG)")

# Sidebar for Uploading PDF
with st.sidebar:
    st.header("Setup Your Data")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    process_button = st.button("Process PDF")

# Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Backend Processing Function
def get_rag_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Persistent database temporary memory mein banayenge
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question ONLY based on the context provided.\n\n"
        "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

# Jab PDF upload ho aur button click ho
if uploaded_file and process_button:
    with st.spinner("Processing PDF... Please wait!"):
        # Save uploaded file locally temporarily
        temp_file = "temp_pdf_storage.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Chain setup karna aur session mein save karna
        st.session_state.rag_chain = get_rag_chain(temp_file)
        st.success("PDF Processed! Ab sawal pucho.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt_input := st.chat_input("ask about your PDF..."):
    if "rag_chain" not in st.session_state:
        st.error("Pehle PDF upload karke 'Process' button dabao!")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke(prompt_input)
                st.markdown(response)
        
        # Add AI message to history
        st.session_state.messages.append({"role": "assistant", "content": response})