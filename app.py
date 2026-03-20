import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

# ─── API Keys ────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="pdfReader_ai",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (Premium Dark UI) ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0F0F1A 0%, #1A0A2E 50%, #0D1117 100%);
}

/* Title area */
.main-title {
    background: linear-gradient(135deg, #7C3AED, #A855F7, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}

.main-subtitle {
    color: #94A3B8;
    font-size: 0.9rem;
    margin-top: 2px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A0A2E 0%, #0F0F1A 100%);
    border-right: 1px solid #2D1B69;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #A855F7;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Cards */
.info-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(168,85,247,0.08));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0;
}

.info-card-title {
    color: #C084FC;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

.info-card-content {
    color: #E2E8F0;
    font-size: 0.88rem;
    line-height: 1.55;
}

/* Summary box */
.summary-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 12px;
    padding: 16px 18px;
    margin: 12px 0;
}

.summary-box h4 {
    color: #34D399;
    margin: 0 0 10px 0;
    font-size: 0.9rem;
}

/* Suggested questions */
.suggestion-pill {
    display: inline-block;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: 5px 14px;
    margin: 4px;
    color: #C084FC;
    font-size: 0.82rem;
    cursor: pointer;
}

/* Chat messages */
.stChatMessage {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    margin: 6px 0;
}

/* Citation badge */
.citation-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.5);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #A5B4FC;
    margin: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A855F7);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #9333EA);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(124,58,237,0.35);
}

/* Chat input */
.stChatInputContainer {
    padding-bottom: 20px;
    background: rgba(255,255,255,0.02);
    border-top: 1px solid rgba(124,58,237,0.2);
}

/* Select box */
.stSelectbox > div > div {
    background: rgba(26,10,46,0.8);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 8px;
    color: #E2E8F0;
}

/* Slider */
.stSlider > div > div > div {
    background: linear-gradient(135deg, #7C3AED, #A855F7);
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 10px;
    padding: 10px;
}

/* Divider */
hr {
    border-color: rgba(124,58,237,0.2);
}

/* Toast */
.stToast {
    background: rgba(26,10,46,0.95);
    border: 1px solid #7C3AED;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #7C3AED !important;
}

/* Badge for doc metadata */
.doc-badge {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 6px;
    padding: 3px 8px;
    font-size: 0.78rem;
    color: #A5B4FC;
    margin-right: 6px;
    margin-bottom: 4px;
    display: inline-block;
}

.stChatInputContainer textarea {
    background: rgba(26,10,46,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
col_title, col_export = st.columns([5, 1])
with col_title:
    st.markdown('<h1 class="main-title">🤖 pdfReader_ai</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Multi-PDF Contextual Chat · Llama 3.3 + Groq · RAG-Powered</p>', unsafe_allow_html=True)

# ─── Cached Resources ────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(temperature=0.1):
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=groq_api_key
    )

# ─── Session State Initialization ───────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "file_list": [],
        "selected_doc": "All Documents",
        "doc_metadata": {},        # {filename: {pages, words, size}}
        "doc_summaries": {},       # {filename: summary_text}
        "suggested_questions": [], # list of question strings
        "temperature": 0.1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─── Helper: Generate Auto-Summary ──────────────────────────────────────────
def generate_summary(docs, filename, llm):
    """Generate a 3-5 bullet point summary of the uploaded document."""
    # Sample first 8000 chars for summary context
    combined_text = "\n".join([d.page_content for d in docs[:15]])[:8000]
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this document in exactly 4-5 concise bullet points. "
        "Focus on the main topics, key arguments, and important facts. "
        "Format each bullet starting with '• '.\n\n"
        "DOCUMENT CONTENT:\n{text}\n\n"
        "SUMMARY (4-5 bullets):"
    )
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({"text": combined_text})

# ─── Helper: Generate Suggested Questions ────────────────────────────────────
def generate_suggested_questions(docs, llm):
    """Generate 3 smart questions a user might ask about this document."""
    combined_text = "\n".join([d.page_content for d in docs[:10]])[:5000]
    q_prompt = ChatPromptTemplate.from_template(
        "Based on this document, generate exactly 3 insightful questions a reader might ask. "
        "Make them specific and interesting — not generic. "
        "Return ONLY the 3 questions, one per line, no numbering, no bullets.\n\n"
        "DOCUMENT:\n{text}\n\nQUESTIONS:"
    )
    chain = q_prompt | llm | StrOutputParser()
    result = chain.invoke({"text": combined_text})
    questions = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return questions[:3]

# ─── Helper: Add PDF to Knowledge Base ──────────────────────────────────────
def add_pdf_to_knowledge(pdf_path, filename):
    embeddings = get_embeddings()
    llm = get_llm(st.session_state.temperature)

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            st.error("❌ No text found in this PDF. It might be a scanned image-only PDF.")
            return False

        # Store metadata
        total_words = sum(len(d.page_content.split()) for d in docs)
        st.session_state.doc_metadata[filename] = {
            "pages": len(docs),
            "words": total_words,
        }

        # Tag each chunk with source filename explicitly
        for doc in docs:
            doc.metadata["filename"] = filename

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        if not splits:
            st.warning("⚠️ Could not extract meaningful text chunks.")
            return False

        # Add to vector store
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
        else:
            st.session_state.vectorstore.add_documents(documents=splits)

        # Generate auto-summary
        with st.spinner(f"📋 Generating summary for **{filename}**..."):
            summary = generate_summary(docs, filename, llm)
            st.session_state.doc_summaries[filename] = summary

        # Generate suggested questions (only for first/latest doc)
        with st.spinner("💡 Generating suggested questions..."):
            questions = generate_suggested_questions(docs, llm)
            st.session_state.suggested_questions = questions

        return True

    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        return False

# ─── Helper: Format Source Citations ─────────────────────────────────────────
def format_docs_with_citations(docs):
    parts = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", doc.metadata.get("filename", "Unknown")))
        page = doc.metadata.get("page", "?")
        parts.append(f"[📄 {source} | Page {int(page)+1 if isinstance(page, int) else page}]\n{doc.page_content}")
    return "\n\n".join(parts)

def get_citations_text(docs):
    seen = set()
    citations = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", doc.metadata.get("filename", "Unknown")))
        page = doc.metadata.get("page", "?")
        page_num = int(page) + 1 if isinstance(page, (int, float)) else page
        key = f"{source}:{page_num}"
        if key not in seen:
            seen.add(key)
            citations.append(f"📄 **{source}** — Page {page_num}")
    return citations

# ─── Helper: Export Chat ─────────────────────────────────────────────────────
def export_chat_as_md():
    lines = [f"# 🤖 pdfReader_ai — Chat Export\n",
             f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             f"**Documents:** {', '.join(st.session_state.file_list) or 'None'}\n",
             "\n---\n"]
    for msg in st.session_state.messages:
        role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"### {role}\n{msg['content']}\n")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Documents")

    if st.session_state.file_list:
        # Document selector
        doc_options = ["All Documents"] + st.session_state.file_list
        try:
            current_index = doc_options.index(st.session_state.selected_doc)
        except ValueError:
            current_index = 0

        st.session_state.selected_doc = st.selectbox(
            "Active Context",
            options=doc_options,
            index=current_index,
            help="Select a document to focus answers on, or 'All Documents' to search across everything."
        )

        # Document metadata
        st.markdown("**📊 Document Stats**")
        for fname in st.session_state.file_list:
            meta = st.session_state.doc_metadata.get(fname, {})
            pages = meta.get("pages", "?")
            words = meta.get("words", "?")
            with st.expander(f"📄 {fname[:28]}{'...' if len(fname)>28 else ''}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pages", pages)
                with col2:
                    st.metric("Words", f"{words:,}" if isinstance(words, int) else words)
                # Show summary if available
                if fname in st.session_state.doc_summaries:
                    st.markdown("**Auto-Summary:**")
                    st.markdown(
                        f'<div class="summary-box"><div style="color:#34D399;font-weight:600;font-size:0.85rem;margin-bottom:8px;">📋 Summary</div>'
                        f'<div style="color:#E2E8F0;font-size:0.83rem;line-height:1.6">{st.session_state.doc_summaries[fname]}</div></div>',
                        unsafe_allow_html=True
                    )
        st.divider()

    # ── AI Settings ──
    st.markdown("### ⚙️ AI Settings")
    st.session_state.temperature = st.slider(
        "🎚️ Creativity (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.05,
        help="0 = Precise & factual | 1 = Creative & exploratory"
    )
    temp_label = "🎯 Precise" if st.session_state.temperature < 0.3 else ("⚖️ Balanced" if st.session_state.temperature < 0.7 else "🎨 Creative")
    st.caption(f"Mode: **{temp_label}**")

    st.divider()

    # ── Memory & Export Controls ──
    st.markdown("### 🛠️ Controls")

    # Export Chat
    if st.session_state.messages:
        chat_export = export_chat_as_md()
        st.download_button(
            label="💾 Export Chat (.md)",
            data=chat_export,
            file_name=f"pdfchat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    # Clear memory
    if st.button("🗑️ Clear All Memory", use_container_width=True):
        for key in ["messages", "vectorstore", "file_list", "selected_doc",
                    "doc_metadata", "doc_summaries", "suggested_questions"]:
            if key in st.session_state:
                del st.session_state[key]
        init_session()
        st.rerun()

    st.divider()

    # ── Tech Stack ──
    st.markdown("### 🚀 Tech Stack")
    st.markdown("""
    <div style="font-size:0.8rem;color:#94A3B8;line-height:1.8">
    ⚡ <b style="color:#A855F7">Groq LPU</b> — Ultra-fast inference<br>
    🧠 <b style="color:#A855F7">Llama 3.3 70B</b> — Language model<br>
    🔍 <b style="color:#A855F7">ChromaDB</b> — Vector search<br>
    🔒 <b style="color:#A855F7">Local Embeddings</b> — Privacy-first<br>
    📚 <b style="color:#A855F7">LangChain RAG</b> — Pipeline
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Vividh Yadav**")
    st.caption("B.Tech AI & ML | BIT Mesra")
    with st.expander("📬 Contact"):
        st.write("📧 vividh50@gmail.com")
        st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/vividh-yadav-866a44380/)")
        st.markdown("[💻 GitHub](https://github.com/VividhDesign)")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Welcome / Empty State ──
if not st.session_state.file_list and not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px 20px; color: #64748B;">
        <div style="font-size: 4rem; margin-bottom: 16px;">📄</div>
        <h2 style="color:#A855F7; font-weight:600; margin-bottom:8px;">Upload a PDF to get started</h2>
        <p style="color:#94A3B8; font-size:0.95rem; max-width:480px; margin:0 auto;">
            Click the <b>📎 paperclip</b> icon in the chat box below to attach a PDF.<br>
            You can upload <b>multiple PDFs</b> and chat across all of them.
        </p>
    </div>
    <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap; margin-top:24px;">
        <div class="info-card" style="max-width:200px; text-align:center;">
            <div style="font-size:1.8rem">⚡</div>
            <div class="info-card-title" style="margin-top:8px">Instant Answers</div>
            <div class="info-card-content" style="font-size:0.8rem">Groq LPU delivers near-zero latency responses</div>
        </div>
        <div class="info-card" style="max-width:200px; text-align:center;">
            <div style="font-size:1.8rem">📍</div>
            <div class="info-card-title" style="margin-top:8px">Page Citations</div>
            <div class="info-card-content" style="font-size:0.8rem">Every answer cites the exact source pages</div>
        </div>
        <div class="info-card" style="max-width:200px; text-align:center;">
            <div style="font-size:1.8rem">🔒</div>
            <div class="info-card-title" style="margin-top:8px">Privacy First</div>
            <div class="info-card-content" style="font-size:0.8rem">Embeddings run locally — your data stays yours</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Render Suggested Questions ────────────────────────────────────────────────
if st.session_state.suggested_questions and st.session_state.vectorstore is not None:
    st.markdown("**💡 Suggested Questions — click to ask:**")
    q_cols = st.columns(len(st.session_state.suggested_questions))
    for i, (col, question) in enumerate(zip(q_cols, st.session_state.suggested_questions)):
        with col:
            if st.button(f"❓ {question}", key=f"suggested_q_{i}", use_container_width=True):
                # Inject this as a user message to process
                st.session_state["_pending_question"] = question
                st.session_state.suggested_questions = []
                st.rerun()

# ── Handle Pending Question from suggestion click ──────────────────────────
pending_question = st.session_state.pop("_pending_question", None)

# ── Render Chat History ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("📍 Sources", expanded=False):
                for cite in message["citations"]:
                    st.markdown(cite)

# ─── Chat Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question or attach a PDF...",
    accept_file="multiple",
    file_type=["pdf"]
)

# ─── Process Input ────────────────────────────────────────────────────────────
def process_question(question_text):
    """Core RAG chain to answer a question and return (response, citations)."""
    llm = get_llm(st.session_state.temperature)

    search_kwargs = {"k": 6, "fetch_k": 20}
    if st.session_state.selected_doc != "All Documents":
        search_kwargs["filter"] = {"source": st.session_state.selected_doc}

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    recent_messages = st.session_state.messages[-6:-1]
    history_text = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}"
        for m in recent_messages
        if "📎" not in m.get("content", "")
    ]) or "No previous history."

    prompt = ChatPromptTemplate.from_template(
        "You are an expert document analyst and intelligent assistant. "
        "Your job is to provide accurate, well-structured answers based ONLY on the provided document context.\n\n"
        "Rules:\n"
        "- Base your answer strictly on the CONTEXT below.\n"
        "- If the answer is not clearly found in the context, say: 'I could not find a direct answer in the document, but based on related content: ...'\n"
        "- Be thorough but concise. Use bullet points for lists.\n"
        "- Do NOT make up information not present in the documents.\n\n"
        "CHAT HISTORY (recent):\n{history}\n\n"
        "DOCUMENT CONTEXT:\n{context}\n\n"
        "USER QUESTION: {input}\n\n"
        "ANSWER:"
    )

    # Get retrieved docs for citations
    retrieved_docs = retriever.invoke(question_text)
    citations = get_citations_text(retrieved_docs)

    def format_docs(docs):
        return format_docs_with_citations(docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "history": lambda x: history_text
        }
        | prompt | llm | StrOutputParser()
    )

    response = rag_chain.invoke(question_text)
    return response, citations


# Handle suggested question click
if pending_question and st.session_state.vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": pending_question})
    with st.chat_message("user"):
        st.markdown(pending_question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, citations = process_question(pending_question)
            st.markdown(response)
            if citations:
                with st.expander("📍 Sources", expanded=True):
                    for cite in citations:
                        st.markdown(cite)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "citations": citations
    })

# Handle main user input
if user_input:
    has_new_file = False

    # 1. Process file uploads
    if user_input.files:
        with st.chat_message("user"):
            file_msg = f"📎 *Uploaded {len(user_input.files)} document(s): {', '.join(f.name for f in user_input.files)}*"
            st.markdown(file_msg)
            st.session_state.messages.append({"role": "user", "content": file_msg})

        for file in user_input.files:
            file_path = file.name
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            with st.status(f"📖 Processing **{file.name}**...", expanded=True) as status:
                st.write("🔍 Extracting text and embedding...")
                success = add_pdf_to_knowledge(file_path, file.name)
                if success:
                    if file.name not in st.session_state.file_list:
                        st.session_state.file_list.append(file.name)
                    st.session_state.selected_doc = file.name
                    has_new_file = True
                    status.update(label=f"✅ **{file.name}** processed!", state="complete")
                else:
                    status.update(label=f"❌ Failed to process {file.name}", state="error")

            if os.path.exists(file_path):
                os.remove(file_path)

        # Show summary in chat as assistant message
        latest_file = st.session_state.file_list[-1] if st.session_state.file_list else None
        if latest_file and latest_file in st.session_state.doc_summaries:
            summary_content = st.session_state.doc_summaries[latest_file]
            meta = st.session_state.doc_metadata.get(latest_file, {})
            assistant_msg = (
                f"📋 **Document Summary — {latest_file}**\n\n"
                f"*Pages: {meta.get('pages','?')} | Words: {meta.get('words',0):,}*\n\n"
                f"{summary_content}"
            )
            with st.chat_message("assistant"):
                st.markdown(assistant_msg)
            st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

    # 2. Process text question
    if user_input.text:
        if st.session_state.vectorstore is None:
            st.warning("📎 Please attach a PDF first using the '+' icon in the chat box.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input.text})
            with st.chat_message("user"):
                st.markdown(user_input.text)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response, citations = process_question(user_input.text)
                    st.markdown(response)
                    if citations:
                        with st.expander("📍 Sources", expanded=True):
                            for cite in citations:
                                st.markdown(cite)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "citations": citations
            })

    # 3. Rerun only for file-only uploads (no text)
    if has_new_file and not user_input.text:
        st.rerun()
