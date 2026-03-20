# 🤖 pdfReader_ai — Professional RAG Chatbot

> **Chat with any PDF** using RAG-powered AI. Faster than ChatGPT. Free. Privacy-first.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-7C3AED?style=for-the-badge&logo=streamlit)](https://pdfreaderai-hgyqrns8rbcwcpfi7wts4i.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/github/license/VividhDesign/pdfReader_ai?style=for-the-badge)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📋 **Auto Document Summary** | Instantly get a 4-5 bullet summary when any PDF is uploaded |
| 📍 **Page-Level Citations** | Every answer cites the exact PDF page it came from |
| 💡 **Suggested Questions** | AI auto-generates 3 smart questions from your document |
| 💾 **Export Chat** | Download entire Q&A session as a Markdown file |
| 🗂️ **Multi-PDF Support** | Upload unlimited PDFs and chat across all of them |
| 🎚️ **Temperature Control** | Adjust AI creativity from precise (0) to creative (1) |
| 📊 **Document Metadata** | See pages, word count for each uploaded PDF |
| ⚡ **Groq LPU Speed** | Near-instant responses via Groq's hardware accelerator |
| 🔒 **Local Embeddings** | Embeddings run on-device — your documents never leave your machine |
| 🎨 **Premium Dark UI** | Polished purple-gradient interface with smooth interactions |

---

## 🆚 Why pdfReader_ai vs ChatGPT / Gemini?

| | pdfReader_ai | ChatGPT Plus | Gemini |
|---|---|---|---|
| **Speed** | ⚡ Groq LPU (fastest) | Medium | Medium |
| **Page Citations** | ✅ Yes | ❌ No | ❌ No |
| **Auto Summary** | ✅ Yes | ❌ No | ❌ No |
| **Privacy** | ✅ Local embeddings | ❌ Cloud | ❌ Cloud |
| **Multi-PDF** | ✅ Unlimited | ⚠️ Limited | ⚠️ Limited |
| **Export Chat** | ✅ Yes | ❌ No | ❌ No |
| **Cost** | 🆓 Free | 💰 $20/mo | Freemium |

---

## 🚀 Tech Stack

- **LLM:** Meta Llama 3.3 70B via [Groq Cloud](https://groq.com) (ultra-low latency LPU)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (runs locally, no API cost)
- **Vector DB:** ChromaDB (in-memory, session-based)
- **Framework:** LangChain (RAG pipeline, MMR retrieval)
- **Frontend:** Streamlit with custom CSS dark theme

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- A free [Groq API Key](https://console.groq.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/VividhDesign/pdfReader_ai.git
cd pdfReader_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 💡 How to Use

1. **Upload a PDF** — Click the 📎 paperclip icon in the chat input
2. **Read the Auto-Summary** — AI summarizes the document for you instantly
3. **Click a Suggested Question** — Or type your own in the chat
4. **See Page Citations** — Expand 📍 Sources to see which pages were used
5. **Export** — Click 💾 Export Chat in the sidebar to save your Q&A

---

## 👨‍💻 Author

**Vividh Yadav** — B.Tech AI & ML, BIT Mesra

- 📧 vividh50@gmail.com
- [LinkedIn](https://www.linkedin.com/in/vividh-yadav-866a44380/)
- [GitHub](https://github.com/VividhDesign)

---

*Built with ❤️ using LangChain, Groq, and Streamlit*
