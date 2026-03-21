# 🤖 pdfReader_ai — Advanced RAG Document Chatbot

> **Chat with any document** using production-grade RAG. Streaming · Re-ranking · Voice · Multi-format · Persistent.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-7C3AED?style=for-the-badge&logo=streamlit)](https://pdfreaderai-hgyqrns8rbcwcpfi7wts4i.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/github/license/VividhDesign/pdfReader_ai?style=for-the-badge)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Streaming Responses** | Tokens appear in real-time like ChatGPT typewriter effect |
| 🎯 **Cross-Encoder Re-ranking** | MMR fetches 10 candidates → re-ranker keeps best 4 |
| 🎙️ **Voice Input** | Groq Whisper voice-to-text transcription |
| 🌐 **Web URL Ingestion** | Add any webpage to your knowledge base |
| 📄 **Multi-Format** | PDF · DOCX · TXT · Web URLs |
| 📋 **Auto Document Summary** | 4-5 bullet summary on every upload |
| 📍 **Page-Level Citations** | Every answer cites exact source pages |
| 💡 **Suggested Questions** | AI auto-generates 3 smart questions |
| 🔀 **Document Compare Mode** | Ask a question, get side-by-side answers from 2 documents |
| 💾 **Export Chat** | Download as `.md` OR styled `.html` report |
| 🗂️ **Multi-Document Support** | Unlimited uploads, cross-document search |
| 💿 **Persistent ChromaDB** | Vector store survives page refresh |
| 📊 **Analytics Panel** | Query count, most cited docs bar chart |
| 🎚️ **Temperature Control** | Adjust AI creativity from precise to creative |
| 🔒 **Local Embeddings** | all-MiniLM-L6-v2 runs on-device — zero data leakage |
| 🎨 **Premium Dark UI** | Purple glassmorphism with CSS animations |

---

## 🏗️ Architecture

```
PDF / DOCX / TXT / URL
  └→ Document Loader (format-aware)
  └→ RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
  └→ HuggingFaceEmbeddings (all-MiniLM-L6-v2) → 384-dim vectors
  └→ ChromaDB.from_documents (persist_directory="./chroma_store")
  └→ Auto-Summary + Suggested Questions (Llama 3.3 70B via Groq)

User Question
  └→ MMR Retriever (fetch_k=25, k=10)
  └→ CrossEncoderReranker (ms-marco-MiniLM-L-6-v2, top_n=4)
  └→ ContextualCompressionRetriever
  └→ ChatPromptTemplate (context + history + question)
  └→ ChatGroq (llama-3.3-70b) → StrOutputParser (streaming)
  └→ st.write_stream() → real-time token display
  └→ Page citations shown in expandable Sources
```

---

## 🚀 Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Llama 3.3 70B via Groq | Free, fast (750 tok/s), open-source |
| Inference | Groq LPU | Specialized chip for LLM inference |
| Voice | Groq Whisper Large v3 Turbo | Fast, free transcription |
| Embeddings | all-MiniLM-L6-v2 (local) | Free, private, 384-dim |
| Retrieval | MMR + CrossEncoderReranker | Diverse + precise |
| Vector DB | ChromaDB (persistent) | Zero setup, survives refresh |
| Orchestration | LangChain LCEL | Composable RAG pipeline |
| Frontend | Streamlit | Native file upload, streaming |

---

## 🛠️ Setup

```bash
# 1. Clone
git clone https://github.com/VividhDesign/pdfReader_ai.git
cd pdfReader_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## 💡 How to Use

1. **Upload** — Click 📎 in chat to attach PDF / DOCX / TXT
2. **Or paste a URL** — Use the sidebar "Add Web URL" field
3. **Or speak** — Click the mic button for voice input
4. **Read Summary** — Auto-generated on upload
5. **Ask Questions** — Click suggestions or type your own
6. **Compare Docs** — Enable Compare Mode in sidebar for side-by-side answers
7. **Export** — Download as `.md` or styled `.html` report

---

## 👨‍💻 Author

**Vividh Yadav** — B.Tech AI & ML, BIT Mesra

- 📧 vividh50@gmail.com
- [LinkedIn](https://www.linkedin.com/in/vividh-yadav-866a44380/)
- [GitHub](https://github.com/VividhDesign)

---

*Built with ❤️ using LangChain LCEL, Groq LPU, ChromaDB, and Streamlit*
