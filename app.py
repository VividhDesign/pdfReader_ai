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
if "file_list" not in st.session_state:
    st.session_state.file_list = []
if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = "All Documents"

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
    embeddings, _ = get_tools()
    
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

        # Add to vector database
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=embeddings
            )
        else:
            st.session_state.vectorstore.add_documents(documents=splits)
            
        return True

    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        return False

# --- SIDEBAR ---
with st.sidebar:
    if st.session_state.file_list:
        st.header("⚙️ Chat Settings")
        
        # 🔥 FIX: Use index to control the selectbox instead of the widget key binding
        doc_options = ["All Documents"] + st.session_state.file_list
        
        try:
            current_index = doc_options.index(st.session_state.selected_doc)
        except ValueError:
            current_index = 0
            
        # Update selected_doc based on the return value of the widget
        st.session_state.selected_doc = st.selectbox(
            "📚 Active Document Context",
            options=doc_options,
            index=current_index
        )
        st.caption("💡 Select 'All Documents' to search across every uploaded PDF.")
        st.divider()

    st.header("⚙️ Memory Control")
    if st.button("🗑️ Clear All Memory", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.file_list = []
        st.session_state.selected_doc = "All Documents"
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
                # Use exact file name so metadata source matches perfectly
                file_path = file.name
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                success = add_pdf_to_knowledge(file_path)
                if success:
                    # Update file list and automatically switch context to the newly uploaded file!
                    if file.name not in st.session_state.file_list:
                        st.session_state.file_list.append(file.name)
                    st.session_state.selected_doc = file.name # This works perfectly now
                    st.toast(f"✅ Processed {file.name}")
                
                # Cleanup temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Rerun to update the sidebar dropdown with the new file
            st.rerun()
                    
    # 2. Agar user ne koi text bheja hai
    if user_input.text:
        if st.session_state.vectorstore is None:
            st.warning("Please attach a PDF using the '+' icon first! 👈")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input.text})
            with st.chat_message("user"):
                st.markdown(user_input.text)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    _, llm = get_tools()
                    
                    # 🔥 FIX: Dynamically filter vectorstore based on user's selection
                    search_kwargs = {"k": 6, "fetch_k": 20}
                    if st.session_state.selected_doc != "All Documents":
                        # Chroma filter syntax: forces AI to only look at the selected file
                        search_kwargs["filter"] = {"source": st.session_state.selected_doc}
                        
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs=search_kwargs
                    )
                    
                    # Get recent chat history for context (last 4 messages)
                    recent_messages = st.session_state.messages[-5:-1]
                    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_messages if '📎' not in m['content']])
                    if not history_text:
                        history_text = "No previous history."

                    prompt = ChatPromptTemplate.from_template(
                        "You are an intelligent assistant. Use the following document extracts and chat history to answer the user's question.\n"
                        "If the user asks about 'this document' or 'the current document', assume they mean the document(s) in the CONTEXT below.\n\n"
                        "RECENT CHAT HISTORY:\n{history}\n\n"
                        "CONTEXT:\n{context}\n\n"
                        "Question: {input}\n"
                        "Answer:"
                    )
                    
                    def format_docs(docs):
                        return "\n\n".join(f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}]\n{doc.page_content}" for doc in docs)

                    # Build chain dynamically at query time
                    rag_chain = (
                        {
                            "context": retriever | format_docs, 
                            "input": RunnablePassthrough(),
                            "history": lambda x: history_text
                        }
                        | prompt | llm | StrOutputParser()
                    )
                    
                    response = rag_chain.invoke(user_input.text)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
