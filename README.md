# Zepto Support Assistant

A GenAI-powered quick-commerce customer support assistant built using **Retrieval-Augmented Generation (RAG)**. The application indexes store policy documents into a ChromaDB vector store, routes queries via a LangGraph state machine, and generates contextual answers using **Kie.ai (GPT-5.2)**.

The frontend is built with **vanilla HTML/JS**, **DaisyUI (Emerald theme)**, and **Tailwind CSS**.

---

## Architecture & Workflow

```text
User Question
      │
      ▼
LangGraph Intent Classifier
      │
      ├─── Policy Question ──► ChromaDB Vector Search ──► Context Extraction ──► Kie.ai GPT-5.2 ──► Answer
      │
      └─── General Question ───────────────────────────────────────────────────► Kie.ai GPT-5.2 ──► Answer
```

---

## Features

* **Deterministic Intent Routing**: Classifies customer queries into policy-specific retrieval (delivery, tracking, refunds, damaged items, cancellations, gift cards, support hours) vs. general inquiries.
* **Vector Store & Embeddings**: Semantic document search using `SentenceTransformer (all-MiniLM-L6-v2)` and persisted `ChromaDB`.
* **Kie.ai LLM Integration**: Reliable responses with custom API error guards and support for offline mock mode (`MOCK_LLM=true`).
* **Modern Chat UI**: DaisyUI Emerald theme with an auto-expanding prompt card, suggested FAQ chips, and Markdown rendering.
* **Modern `uv` Project**: Fast, reproducible dependency resolution with `pyproject.toml` and cross-platform `uv.lock`.

---

## Quickstart with `uv` (Recommended)

This project uses [Astral `uv`](https://docs.astral.sh/uv/) for Python dependency management.

### 1. Clone & Setup Environment

```bash
git clone https://github.com/sakshiipandey/zepto-support-assistant.git
cd zepto-support-assistant

# Sync exact pinned dependencies in seconds
uv sync
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your Kie.ai API key:
```env
KIE_API_KEY=your_kie_ai_api_key_here
OPENAI_BASE_URL=https://api.kie.ai/gpt-5-2/v1
LLM_MODEL=gpt-5-2

# Optional: set to true to test locally without consuming API credits
MOCK_LLM=false
```

### 3. (Optional) Index Knowledge Documents

The pre-indexed vector store is included, but you can re-index anytime:
```bash
uv run python load_documents.py
```

### 4. Run Application

```bash
uv run python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Docker Deployment (Optional)

Docker is completely optional for local development with `uv`, but container files are provided for cloud hosting or isolated deployments:

```bash
docker compose up --build
```

---

## Project Structure

```text
zepto-support-assistant/
│
├── docs/                 # Policy documents (delivery, refunds, tracking, etc.)
├── static/               # CSS styles augmenting DaisyUI
├── templates/            # index.html (vanilla HTML + DaisyUI Emerald)
├── vector_store/         # ChromaDB persistent vector database
│
├── app.py                # Flask web server & /ask API endpoint
├── graph.py              # LangGraph workflow & intent classification
├── rag.py                # Retrieval logic, ChromaDB query, & Kie.ai client
├── load_documents.py     # Document chunking & vector embedding script
│
├── pyproject.toml        # Standard Python project metadata & uv CPU-torch config
├── uv.lock               # Pinned, reproducible dependency lockfile
├── requirements.txt      # Legacy pip requirements (for backward compatibility)
├── Dockerfile            # Optimized multi-stage Docker build using uv
├── docker-compose.yml    # Docker compose specification
└── .env.example          # Template for environment configuration
```