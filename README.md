# 🤖 pdfReader_ai: Professional RAG Chatbot

An advanced Retrieval-Augmented Generation (RAG) application that allows users to have natural conversations with any PDF document. 

## 🚀 Technical Stack
- **LLM:** Meta Llama 3.3 (via Groq Cloud for ultra-low latency)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local Execution)
- **Vector Database:** ChromaDB
- **Framework:** LangChain
- **Frontend:** Streamlit

## ✨ Key Features
- **Instant Processing:** Leverages Groq's LPU for near-instant responses.
- **Local Embeddings:** Optimized for cost and privacy by running embeddings locally.
- **Memory-Efficient:** Implemented Streamlit caching for faster resource loading.

## 🛠️ Installation & Setup
1. Clone the repo: `git clone https://github.com/YOUR_USER/pdfReader_ai.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API Keys in `.env`
4. Run: `streamlit run app.py`
