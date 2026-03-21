import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime

try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_OK = True
except Exception:
    DOCX_OK = False

try:
    from langchain_community.document_loaders import TextLoader
    TXT_OK = True
except Exception:
    TXT_OK = False

load_dotenv()
PERSIST_DIR = "./chroma_store"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="pdfReader_ai",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load CSS ────────────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(_css_path):
    with open(_css_path, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
col_t, col_e = st.columns([5, 1])
with col_t:
    st.markdown('<h1 class="main-title">🤖 pdfReader_ai</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">Multi-Format Chat · Llama 3.3 70B + Groq · '
        'Streaming · Re-ranking · Voice · Persistent Memory</p>',
        unsafe_allow_html=True,
    )

# ─── Cached Resources ─────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_reranker():
    m = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=m, top_n=4)

def get_llm(temp=0.1):
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=temp,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

# ─── Session State ────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "vectorstore": None,
        "file_list": [],
        "selected_doc": "All Documents",
        "doc_metadata": {},
        "doc_summaries": {},
        "suggested_questions": [],
        "temperature": 0.1,
        "citation_counts": {},
        "query_count": 0,
        "compare_mode": False,
        "compare_docs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Load persisted vectorstore
    if st.session_state.vectorstore is None and os.path.exists(PERSIST_DIR):
        try:
            vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
            st.session_state.vectorstore = vs
            if not st.session_state.file_list:
                data = vs.get()
                srcs = {
                    m["filename"]
                    for m in data.get("metadatas", [])
                    if m and "filename" in m
                }
                st.session_state.file_list = sorted(srcs)
        except Exception:
            st.session_state.vectorstore = None

init_session()

# ─── Loaders ─────────────────────────────────────────────────────────────────
def load_document(path, name):
    ext = os.path.splitext(name)[1].lower()
    try:
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        if ext == ".docx" and DOCX_OK:
            return Docx2txtLoader(path).load()
        if ext == ".txt" and TXT_OK:
            return TextLoader(path, encoding="utf-8").load()
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {e}")
    return None

def load_url(url):
    try:
        docs = WebBaseLoader(url).load()
        for d in docs:
            d.metadata["filename"] = url
            d.metadata["source"] = url
        return docs
    except Exception as e:
        st.error(f"❌ Failed to load URL: {e}")
        return None

# ─── Ingestion ────────────────────────────────────────────────────────────────
def ingest_documents(docs, filename):
    embeddings = get_embeddings()
    llm = get_llm(st.session_state.temperature)
    total_words = sum(len(d.page_content.split()) for d in docs)
    doc_type = os.path.splitext(filename)[1].upper().lstrip(".") or "WEB"
    st.session_state.doc_metadata[filename] = {
        "pages": len(docs), "words": total_words, "type": doc_type,
    }
    for d in docs:
        d.metadata["filename"] = filename
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    splits = splitter.split_documents(docs)
    if not splits:
        st.warning("⚠️ Could not extract text.")
        return False
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=PERSIST_DIR,
        )
    else:
        st.session_state.vectorstore.add_documents(splits)
    with st.spinner(f"📋 Summarizing **{filename}**..."):
        st.session_state.doc_summaries[filename] = _gen_summary(docs, llm)
    with st.spinner("💡 Generating questions..."):
        st.session_state.suggested_questions = _gen_questions(docs, llm)
    return True

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _gen_summary(docs, llm):
    text = "\n".join([d.page_content for d in docs[:15]])[:8000]
    p = ChatPromptTemplate.from_template(
        "Summarize in 4-5 concise bullet points starting with '• '.\n\n"
        "DOCUMENT:\n{text}\n\nSUMMARY:"
    )
    return (p | llm | StrOutputParser()).invoke({"text": text})

def _gen_questions(docs, llm):
    text = "\n".join([d.page_content for d in docs[:10]])[:5000]
    p = ChatPromptTemplate.from_template(
        "Generate exactly 3 specific insightful questions. One per line, no numbering.\n\n"
        "DOCUMENT:\n{text}\n\nQUESTIONS:"
    )
    result = (p | llm | StrOutputParser()).invoke({"text": text})
    return [q.strip() for q in result.strip().split("\n") if q.strip()][:3]

def _citations(docs):
    seen, out = set(), []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", d.metadata.get("filename", "Unknown")))
        pg = d.metadata.get("page", "?")
        pn = int(pg) + 1 if isinstance(pg, (int, float)) else pg
        key = f"{src}:{pn}"
        if key not in seen:
            seen.add(key)
            out.append(f"📄 **{src}** — Page {pn}")
    return out

def _fmt_docs(docs):
    parts = []
    for d in docs:
        src = os.path.basename(d.metadata.get("source", d.metadata.get("filename", "?")))
        pg = d.metadata.get("page", "?")
        pn = int(pg) + 1 if isinstance(pg, (int, float)) else pg
        parts.append(f"[Source: {src} | Page {pn}]\n{d.page_content}")
    return "\n\n".join(parts)

# ─── RAG Chain (Streaming) ────────────────────────────────────────────────────
def _make_retriever(doc_filter=None):
    sk = {"k": 10, "fetch_k": 25}
    if doc_filter:
        sk["filter"] = {"filename": doc_filter}
    base = st.session_state.vectorstore.as_retriever(search_type="mmr", search_kwargs=sk)
    return ContextualCompressionRetriever(base_compressor=get_reranker(), base_retriever=base)

def stream_answer(question, doc_filter=None):
    llm = get_llm(st.session_state.temperature)
    retriever = _make_retriever(doc_filter)
    # Get raw docs for citations (fast path, no reranker overhead)
    sk2 = {"k": 6}
    if doc_filter:
        sk2["filter"] = {"filename": doc_filter}
    raw_docs = st.session_state.vectorstore.similarity_search(question, **sk2)
    citations = _citations(raw_docs)
    for d in raw_docs:
        fn = d.metadata.get("filename", "Unknown")
        st.session_state.citation_counts[fn] = st.session_state.citation_counts.get(fn, 0) + 1
    history = "\n".join([
        f"{m['role'].upper()}: {m['content'][:300]}"
        for m in st.session_state.messages[-6:-1]
        if "📎" not in m.get("content", "")
    ]) or "No prior conversation."
    prompt = ChatPromptTemplate.from_template(
        "You are an expert document analyst.\n"
        "Rules:\n"
        "- Answer ONLY from DOCUMENT CONTEXT.\n"
        "- Be thorough, structured; use bullet points for lists.\n"
        "- If not found: 'I could not find a direct answer, but based on related content: ...'\n"
        "- Never fabricate information.\n\n"
        "HISTORY:\n{history}\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {input}\n\nANSWER:"
    )
    chain = (
        {
            "context": retriever | _fmt_docs,
            "input": RunnablePassthrough(),
            "history": lambda x: history,
        }
        | prompt | llm | StrOutputParser()
    )
    return chain.stream(question), citations

# ─── Voice Transcription ──────────────────────────────────────────────────────
def transcribe_audio(audio_bytes):
    try:
        from groq import Groq as GC
        client = GC(api_key=os.getenv("GROQ_API_KEY"))
        buf = io.BytesIO(audio_bytes)
        buf.name = "recording.wav"
        return client.audio.transcriptions.create(
            model="whisper-large-v3-turbo", file=buf, language="en"
        ).text
    except Exception as e:
        st.error(f"❌ Transcription failed: {e}")
        return None

# ─── Exports ──────────────────────────────────────────────────────────────────
def _export_html():
    msgs = ""
    for m in st.session_state.messages:
        role = "🧑 You" if m["role"] == "user" else "🤖 Assistant"
        color = "#7C3AED" if m["role"] == "user" else "#10B981"
        content = m["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        msgs += (
            f'<div style="margin:16px 0;padding:16px;background:rgba(255,255,255,.04);'
            f'border-left:3px solid {color};border-radius:8px">'
            f'<div style="color:{color};font-weight:600;margin-bottom:8px">{role}</div>'
            f'<div style="color:#E2E8F0;line-height:1.7">{content}</div></div>'
        )
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    docs = ", ".join(st.session_state.file_list) or "None"
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>pdfReader_ai Export</title>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
        '<style>body{font-family:Inter,sans-serif;background:#0F0F1A;color:#E2E8F0;'
        'max-width:800px;margin:0 auto;padding:40px 24px}'
        'h1{background:linear-gradient(135deg,#7C3AED,#A855F7);-webkit-background-clip:text;'
        '-webkit-text-fill-color:transparent;font-size:2rem}'
        '.meta{color:#94A3B8;font-size:.85rem;margin-bottom:32px}</style></head>'
        f'<body><h1>🤖 pdfReader_ai Export</h1>'
        f'<div class="meta">Exported: {ts} | Documents: {docs}</div>'
        f'{msgs}</body></html>'
    )

def _export_md():
    lines = [
        "# 🤖 pdfReader_ai — Chat Export",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Documents:** {', '.join(st.session_state.file_list) or 'None'}",
        "\n---\n",
    ]
    for m in st.session_state.messages:
        role = "🧑 You" if m["role"] == "user" else "🤖 Assistant"
        lines.append(f"### {role}\n{m['content']}\n")
    return "\n".join(lines)

# ─── Run Question ─────────────────────────────────────────────────────────────
def run_question(question, doc_filter=None):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        gen, citations = stream_answer(question, doc_filter)
        response = st.write_stream(gen)
        if citations:
            with st.expander("📍 Sources", expanded=True):
                for c in citations:
                    st.markdown(c)
    st.session_state.messages.append({
        "role": "assistant", "content": response, "citations": citations,
    })
    st.session_state.query_count += 1

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Documents")
    if st.session_state.file_list:
        opts = ["All Documents"] + st.session_state.file_list
        try:
            idx = opts.index(st.session_state.selected_doc)
        except ValueError:
            idx = 0
        st.session_state.selected_doc = st.selectbox("Active Context", opts, index=idx)
        st.markdown("**📊 Document Stats**")
        for fname in st.session_state.file_list:
            meta = st.session_state.doc_metadata.get(fname, {})
            icon = "🌐" if "://" in fname else "📄"
            short = fname[:26] + "..." if len(fname) > 26 else fname
            with st.expander(f"{icon} {short}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Pages", meta.get("pages", "?"))
                with c2:
                    w = meta.get("words", 0)
                    st.metric("Words", f"{w:,}" if isinstance(w, int) else w)
                if fname in st.session_state.doc_summaries:
                    st.markdown(
                        '<div class="summary-box">'
                        '<div style="color:#34D399;font-weight:600;font-size:.85rem;margin-bottom:8px">📋 Summary</div>'
                        f'<div style="color:#E2E8F0;font-size:.83rem;line-height:1.6">{st.session_state.doc_summaries[fname]}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
        st.divider()

    st.markdown('### 🌐 Add Web URL <span class="new-badge">NEW</span>', unsafe_allow_html=True)
    url_val = st.text_input("URL", placeholder="https://...", label_visibility="collapsed")
    if st.button("➕ Ingest URL", use_container_width=True, disabled=not url_val):
        with st.status("🌐 Loading...", expanded=True) as status:
            st.write("Fetching webpage content...")
            docs = load_url(url_val)
            if docs:
                ok = ingest_documents(docs, url_val)
                if ok:
                    if url_val not in st.session_state.file_list:
                        st.session_state.file_list.append(url_val)
                    st.session_state.selected_doc = url_val
                    status.update(label="✅ URL ingested!", state="complete")
                    st.rerun()
    st.divider()

    st.markdown('### 🔀 Compare Mode <span class="new-badge">NEW</span>', unsafe_allow_html=True)
    compare_on = st.toggle("Enable Document Comparison", value=st.session_state.compare_mode)
    st.session_state.compare_mode = compare_on
    if compare_on:
        if len(st.session_state.file_list) >= 2:
            sel = st.multiselect(
                "Pick 2 docs",
                st.session_state.file_list,
                default=st.session_state.compare_docs[:2] if st.session_state.compare_docs else [],
                max_selections=2,
            )
            st.session_state.compare_docs = sel
            if len(sel) == 2:
                st.success("✅ Compare mode ready!")
        else:
            st.info("Upload at least 2 documents.")
    st.divider()

    st.markdown("### ⚙️ AI Settings")
    st.session_state.temperature = st.slider(
        "🎚️ Creativity", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="0 = Precise | 1 = Creative",
    )
    lbl = (
        "🎯 Precise" if st.session_state.temperature < 0.3
        else "⚖️ Balanced" if st.session_state.temperature < 0.7
        else "🎨 Creative"
    )
    st.caption(f"Mode: **{lbl}**")
    st.divider()

    st.markdown('### 📊 Analytics <span class="new-badge">NEW</span>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1:
        st.metric("Queries", st.session_state.query_count)
    with a2:
        st.metric("Docs", len(st.session_state.file_list))
    if st.session_state.citation_counts:
        st.markdown("**📈 Most Cited:**")
        st.bar_chart(st.session_state.citation_counts, height=100)
    st.divider()

    st.markdown("### 🛠️ Controls")
    if st.session_state.messages:
        b1, b2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        with b1:
            st.download_button(
                "💾 .md", _export_md(), f"chat_{ts}.md", "text/markdown",
                use_container_width=True,
            )
        with b2:
            st.download_button(
                "🌐 .html", _export_html(), f"chat_{ts}.html", "text/html",
                use_container_width=True,
            )
    if st.button("🗑️ Clear Session", use_container_width=True):
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        init_session()
        st.rerun()
    st.divider()

    st.markdown("### 🚀 Tech Stack")
    st.markdown(
        '<div style="font-size:0.8rem;color:#94A3B8;line-height:1.8">'
        '⚡ <b style="color:#A855F7">Groq LPU</b> — Streaming responses<br>'
        '🧠 <b style="color:#A855F7">Llama 3.3 70B</b> — Language model<br>'
        '🎙️ <b style="color:#A855F7">Groq Whisper</b> — Voice transcription<br>'
        '🎯 <b style="color:#A855F7">Cross-Encoder</b> — Re-ranking retrieval<br>'
        '🔍 <b style="color:#A855F7">ChromaDB</b> — Persistent vector DB<br>'
        '🔒 <b style="color:#A855F7">Local Embeddings</b> — Privacy-first<br>'
        '📚 <b style="color:#A855F7">LangChain LCEL</b> — RAG pipeline'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 👨‍💻 Developed by")
    st.markdown("**Vividh Yadav**")
    st.caption("B.Tech AI & ML | BIT Mesra")
    with st.expander("📬 Contact"):
        st.write("📧 vividh50@gmail.com")
        st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/vividh-yadav-866a44380/)")
        st.markdown("[💻 GitHub](https://github.com/VividhDesign)")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════════════════

# Welcome screen
if not st.session_state.file_list and not st.session_state.messages:
    st.markdown(
        '<div style="text-align:center;padding:40px 20px">'
        '<div style="font-size:4rem;margin-bottom:16px">📄</div>'
        '<h2 style="color:#A855F7;font-weight:600;margin-bottom:8px">'
        'Upload any document or paste a URL</h2>'
        '<p style="color:#94A3B8;font-size:.95rem;max-width:560px;margin:0 auto">'
        'Supports <b>PDF · DOCX · TXT · Web URLs</b><br>'
        'Click the 📎 icon in the chat box to attach a file'
        '</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="info-card" style="text-align:center"><div style="font-size:1.8rem">⚡</div>'
            '<div class="info-card-title" style="margin-top:8px">Streaming</div>'
            '<div class="info-card-content" style="font-size:.8rem">Tokens in real-time</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card" style="text-align:center"><div style="font-size:1.8rem">🎯</div>'
            '<div class="info-card-title" style="margin-top:8px">Re-ranking</div>'
            '<div class="info-card-content" style="font-size:.8rem">Cross-encoder precision</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card" style="text-align:center"><div style="font-size:1.8rem">🎙️</div>'
            '<div class="info-card-title" style="margin-top:8px">Voice Input</div>'
            '<div class="info-card-content" style="font-size:.8rem">Groq Whisper</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="info-card" style="text-align:center"><div style="font-size:1.8rem">🌐</div>'
            '<div class="info-card-title" style="margin-top:8px">Web URLs</div>'
            '<div class="info-card-content" style="font-size:.8rem">Ingest any webpage</div></div>',
            unsafe_allow_html=True,
        )

# Suggested Questions
if st.session_state.suggested_questions and st.session_state.vectorstore:
    st.markdown("**💡 Suggested Questions — click to ask:**")
    sq_cols = st.columns(len(st.session_state.suggested_questions))
    for i, (col, q) in enumerate(zip(sq_cols, st.session_state.suggested_questions)):
        with col:
            if st.button(f"❓ {q}", key=f"sq_{i}", use_container_width=True):
                st.session_state["_pending_q"] = q
                st.session_state.suggested_questions = []
                st.rerun()

pending_q = st.session_state.pop("_pending_q", None)

# Voice Input
if st.session_state.vectorstore is not None:
    st.markdown("**🎙️ Voice Input — click mic to speak your question:**")
    try:
        audio = st.audio_input("Record question", label_visibility="collapsed", key="voice_in")
        if audio:
            with st.spinner("🎙️ Transcribing via Groq Whisper..."):
                text = transcribe_audio(audio.read())
            if text:
                st.info(f"🎙️ Heard: *{text}*")
                st.session_state["_pending_q"] = text
                st.rerun()
    except AttributeError:
        st.caption("🎙️ Voice input requires Streamlit ≥ 1.40. Run: `pip install --upgrade streamlit`")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📍 Sources", expanded=False):
                for c in msg["citations"]:
                    st.markdown(c)

# Handle pending question (voice/suggestion)
if pending_q and st.session_state.vectorstore is not None:
    df = (
        st.session_state.selected_doc
        if st.session_state.selected_doc != "All Documents"
        else None
    )
    run_question(pending_q, df)

# ─── Chat Input ──────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask a question, or attach PDF / DOCX / TXT...",
    accept_file="multiple",
    file_type=["pdf", "docx", "txt"],
)

if user_input:
    # Handle file uploads
    if user_input.files:
        with st.chat_message("user"):
            names = ", ".join(f.name for f in user_input.files)
            msg = f"📎 *Uploaded {len(user_input.files)} file(s): {names}*"
            st.markdown(msg)
            st.session_state.messages.append({"role": "user", "content": msg})

        for file in user_input.files:
            fpath = file.name
            with open(fpath, "wb") as f:
                f.write(file.getvalue())

            with st.status(f"📖 Processing **{file.name}**...", expanded=True) as status:
                st.write("🔍 Loading, chunking and embedding...")
                docs = load_document(fpath, file.name)
                if docs is None:
                    status.update(label=f"❌ Unsupported format: {file.name}", state="error")
                else:
                    ok = ingest_documents(docs, file.name)
                    if ok:
                        if file.name not in st.session_state.file_list:
                            st.session_state.file_list.append(file.name)
                        st.session_state.selected_doc = file.name
                        status.update(label=f"✅ **{file.name}** ready!", state="complete")

            if os.path.exists(fpath):
                os.remove(fpath)

        latest = st.session_state.file_list[-1] if st.session_state.file_list else None
        if latest and latest in st.session_state.doc_summaries:
            meta = st.session_state.doc_metadata.get(latest, {})
            smsg = (
                f"📋 **Summary — {latest}**\n\n"
                f"*Pages: {meta.get('pages', '?')} | Words: {meta.get('words', 0):,}*\n\n"
                f"{st.session_state.doc_summaries[latest]}"
            )
            with st.chat_message("assistant"):
                st.markdown(smsg)
            st.session_state.messages.append({"role": "assistant", "content": smsg})

    # Handle text question
    if user_input.text:
        if st.session_state.vectorstore is None:
            st.warning("📎 Please attach a document or add a URL first.")
        else:
            q = user_input.text
            # Compare mode
            if st.session_state.compare_mode and len(st.session_state.compare_docs) == 2:
                d1, d2 = st.session_state.compare_docs
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                st.markdown(
                    f'<div class="compare-header">🔀 Comparing: '
                    f'<b>{os.path.basename(d1)}</b> vs <b>{os.path.basename(d2)}</b></div>',
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📄 {os.path.basename(d1)}**")
                    with st.chat_message("assistant"):
                        g1, _ = stream_answer(q, d1)
                        r1 = st.write_stream(g1)
                with col2:
                    st.markdown(f"**📄 {os.path.basename(d2)}**")
                    with st.chat_message("assistant"):
                        g2, _ = stream_answer(q, d2)
                        r2 = st.write_stream(g2)
                combined = (
                    f"**[Compare] {q}**\n\n"
                    f"**{os.path.basename(d1)}:**\n{r1}\n\n"
                    f"**{os.path.basename(d2)}:**\n{r2}"
                )
                st.session_state.messages.append({"role": "assistant", "content": combined})
                st.session_state.query_count += 1
            else:
                df = (
                    st.session_state.selected_doc
                    if st.session_state.selected_doc != "All Documents"
                    else None
                )
                run_question(q, df)

    if user_input.files and not user_input.text:
        st.rerun()
