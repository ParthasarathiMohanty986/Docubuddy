# Local PDF RAG chatbot

Streamlit + Chroma + Ollama. Upload one PDF, ask questions, and inspect supporting passages with page numbers. No paid API or key is needed.

## Setup (Windows, Python 3.11 / 3.12)

Install Ollama from https://ollama.com/download/windows and ensure it is running. Download these models only if not already installed:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:1.7b
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Start

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501, upload a text PDF, and ask a question. Defaults: local Ollama at `http://localhost:11434`, `embeddinggemma:latest` for embeddings, and `qwen3:1.7b` for answers. You may select `qwen3:4b` if installed, but it needs more memory.

If Ollama is not running, launch it or run `ollama serve` in another terminal. Use `ollama list` to check models. Search passages only skips answer generation but still uses Ollama embeddings.

## How it works

1. pypdf extracts text separately for each page.
2. Text is split into 800-character chunks with 120-character overlap.
3. Ollama embeddinggemma converts chunks into vectors.
4. Chroma stores vectors, text, and page numbers in memory.
5. The same embedding model encodes the question. Chroma retrieves four closest passages.
6. Qwen receives the question and passages, with instructions to cite pages or acknowledge missing information.

Both models run locally at the default URL. Changing that URL sends text to the selected server. Thinking is disabled and answers are capped at 512 tokens. Optional environment variables: `OLLAMA_HOST` (HTTP URL), `LLM_MODEL`.

## Demo limits

- One text PDF, up to 20 MB / 200 pages. No OCR; tables may extract poorly.
- Questions are independent. Displayed history is not used for retrieval.
- Character chunks can split sentences. Page references are physical PDF page positions starting at 1.
- Retrieved sources are evidence, not verified citations. Models can make mistakes. Nearest matches are returned even for unanswerable questions.
- Indexes are in memory. Replacing/removing a PDF deletes its active collection. Restarting clears all indexes. Disconnected sessions may leave collections in memory until restart.
- Local demo only: no authentication or production upload isolation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover PDF extraction, empty documents, chunk boundaries, Chroma retrieval, and answer requests without model downloads.

References: [Ollama embeddings](https://docs.ollama.com/api/embed), [Ollama chat](https://docs.ollama.com/api/chat), [Chroma](https://docs.trychroma.com/reference/python).
