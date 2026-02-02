# MVP Project: Chat With Your Docs
A conversational RAG application that answers questions from your uploaded documents (PDF, TXT, DOC). 

## a. Quick setup

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (free local LLM)
- Pull a model: `ollama pull qwen2:latest`
- Run the model locally with: `ollama run qwen2:latest`

### 2. How to run project locally
#### Create Virtual env 
   `python -m venv .venv`
   `.venv\Scripts\activate`

#### Install
`pip install -r requirements.txt`

#### Optional: copy env and edit
`copy .env.example .env`
#### Set OLLAMA_BASE_URL if Ollama is not on localhost
>
#### Terminal 1 - Start backend - API
`uvicorn app.main:app --reload`

#### Terminal 2 - start UI
`streamlit run ui/streamlit_app.py`

---
## b. Architecture overview
##`High-level flow
1. User (Streamlit UI) uploads files and asks questions.
2. Streamlit → FastAPI calls:
   - POST /api/upload to ingest documents
   - POST /api/ask (or GET /api/ask/stream) to query
3. Ingestion pipeline (FastAPI):
   - Loader by type (PDF/TXT/DOC/DOCX)
   - Chunking with RecursiveCharacterTextSplitter (800 / 100)
   - Store chunks + metadata in ChromaDB (persistent local store)
4. Retrieval pipeline (FastAPI) on each question:
   - Semantic retrieval: Chroma similarity search (top 5)
   - Keyword retrieval: BM25 over all stored chunks (top 5)
   - Fusion: Reciprocal Rank Fusion (RRF) → final top 5 context chunks
5. LLM answering (FastAPI):
   - Build prompt = system instruction + retrieved context + user question
   - Call chat model
   - Return answer + optional source snippets; streaming endpoint streams tokens

                    ┌──────────────────────────────────────────────┐
                    │                 Client Layer                 │
                    │           Streamlit UI (port 8501)           │
                    │  - Upload files                              │
                    │  - Chat + session memory                     │
                    │  - Source toggle                             │
                    └───────────────┬──────────────────────────────┘
                                    │ HTTP
                                    │ /api/upload  /api/ask  /api/ask/stream  /api/collections
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           API / Orchestration Layer                          │
│                         FastAPI Backend (port 8000)                          │
│  Cross-cutting: trace_id, structured logs, latency metrics, error handling   │
│  Guardrails: prompt-injection check, question length limit, context cap      │
└───────────────┬───────────────────────────────────────────────┬──────────────┘
                │                                               │
                │ Upload path                                   │ Query path
                ▼                                               ▼
┌──────────────────────────────-─┐                 ┌───────────────────────────────┐
│ Ingestion Service              │                 │ Retrieval + QA Service        │
│ - File loaders (PDF/TXT/DOCX)  │                 │ - Semantic search (Chroma)    │
│ - Chunking (800/100)           │                 │   top_k=5                     │
│ - Add metadata (filename/page/ │                 │ - BM25 search (in-memory)     │
│   upload_time)                 │                 │   built from Chroma docs      │
│ - Embed + upsert to Chroma     │                 │ - Merge (RRF) → final top_k=5 │
└───────────────┬──────────────-─┘                 └───────────────┬───────────────┘
                │                                                 │
                ▼                                                 ▼
      ┌─────────────────────-──┐                      ┌───────────────────────────┐
      │ Storage Layer          │                      │ Prompt + LLM Layer        │
      │ ChromaDB (persistent)  │◄────────────────────►│ - Prompt: system + context│
      │ - chunk text + metadata│   fetch chunk texts  │   + question              │
      │ - embeddings           │   for BM25 index     │ - Stream tokens back      │
      └──────────────────────-─┘                      └───────────────────────────┘

   

```

- **UI**: Streamlit — file upload, chat, source references, session conversation memory.
- **Backend**: FastAPI — `/upload`, `/ask`, `/collections`, `/collections/{name}` (DELETE).
- **RAG**: LangChain — loaders, splitter, retriever, prompt, LLM.
- **Retrieval**: Hybrid — semantic (ChromaDB + sentence-transformers) + BM25; merge with RRF; **top 5** results.
- **LLM**: Ollama (free, local) — e.g. `qwen2:latest`.
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`) — free, no API key.

---

## c. Productionization (AWS / GCP / Azure)

What would be required to make this scalable and deployable on a hyper-scaler:

1. **Compute**
   - Run FastAPI and Streamlit as separate services (e.g. ECS/Fargate, Cloud Run, or App Service).
   - Optionally move to a serverless pattern (Lambda + API Gateway) with smaller, stateless handlers; keep long-running RAG in a managed service.

2. **Vector DB**
   - Replace single-node Chroma with a managed vector store: **Amazon OpenSearch Serverless** (with k-NN), **Pinecone**, **Weaviate**, or **pgvector** (RDS/Aurora) so indexing and search scale and persist across restarts.

3. **LLM**
   - Keep Ollama for cost-free dev; for production, switch to **OpenAI**, **Anthropic**, or **AWS Bedrock** with API keys in Secrets Manager / Parameter Store.
   - Another option to use vLLM for LLM inference and serving to get better performance

4. **Embeddings**
   - Keep sentence-transformers in the app, or use a managed embedding API (OpenAI, Bedrock) for consistency and less GPU/CPU on your side.

5. **Storage**
   - Store original files in **S3 / GCS / Blob**; trigger ingestion via events (Lambda, Cloud Functions) or a queue (SQS) for async processing.

6. **Auth & multi-tenancy**
   - Add auth (Cognito, OAuth2) and scope collections/indexes by user or tenant; enforce rate limits and quotas.

7. **Observability**
   - Send logs and metrics to **CloudWatch / Datadog / Prometheus**; add tracing (X-Ray, OpenTelemetry) and token-usage logging for cost and quality.

8. **CI/CD**
   - Deploy with **CodePipeline**, **GitHub Actions**, or **Azure DevOps**; run tests in CI.

---

## d. RAG / LLM approach and decisions

- **LLM → Ollama**
We used Ollama to run the language model locally because it is free, does not require an API key, and works well for assignments and demos.

- **Embeddings → sentence-transformers (all-MiniLM-L6-v2)**
This model is free to use and provides a good balance between speed and accuracy for converting text into vector embeddings.

- **Vector Database → ChromaDB**
ChromaDB stores embeddings locally in a persistent way. It is simple to use and integrates well with LangChain.

- **Framework → LangChain**
LangChain was chosen to manage the RAG pipeline because it provides ready-made tools for document loading, retrieval, and chaining LLM responses.

- **Retrieval Method → Hybrid Search (semantic + BM25, top 5 results)**
We combined semantic search with keyword-based BM25 and used Reciprocal Rank Fusion (RRF) to get more accurate and balanced search results as required in the assignment.

- **Chunking → RecursiveCharacterTextSplitter** with `chunk_size=800`, `chunk_overlap=100`.
Keeps context across chunk boundaries; 800/100 is a reasonable default for short answers and source citations.

- **Embeddings → all-MiniLM-L6-v2**
384 dims, fast, good for semantic similarity on documents.

- **Retrieval** →
Semantic: ChromaDB similarity search (top 5).
BM25: In-memory index over the same chunks (built from Chroma when needed); top 5.
Merge: Reciprocal Rank Fusion (RRF) then take **top 5** for the prompt.

- **Prompts** →
System: “Answer only from the provided context; if not in context, say so; cite source when possible.”
User: Retrieved context + user question.
Keeps the model grounded and reduces hallucination.

- **Guardrails** →
Prompt injection: Simple deny-list of phrases (e.g. “ignore previous instructions”); reject and return a safe message.
Max tokens: Capped in Ollama config (`num_predict` / `max_answer_tokens`).
Relevance: Optional next step — score or filter retrieved chunks by similarity threshold before building context (not implemented in this minimal version).

- **Observability**
Logging: Structured logs with optional `x-trace-id`; log upload/ask latency and errors.
Token usage: Not logged in this version; would add from Ollama/OpenAI response when moving to production.

---

## e. Key technical decisions

1. **Hybrid search (semantic + BM25)**  
   Matches the brief and improves results when users use exact terms (BM25) vs. paraphrased questions (semantic).

2. **Single default collection**  
   One Chroma collection keeps the demo simple; multi-collection (e.g. per user) is a config/API extension.

3. **BM25 from Chroma**  
   BM25 index is built from Chroma’s stored documents when needed (no separate doc store). Fine for small/medium corpora; for large scale, a dedicated search index (e.g. OpenSearch) would be better.

4. **Sync RAG chain, async streaming optional**  
   Main `/ask` is synchronous; streaming endpoint is present for future use. Keeps the code simple while allowing async where it matters.

5. **Streamlit for UI**  
   Fast to build and good enough to demonstrate upload, chat, and source refs; production would likely use a proper frontend (React, etc.) talking to the same API.

---

## f. Engineering standards

- **Type hints** on public functions and API models.
- **Pydantic** for request/response and config (no hard-coded secrets; config from env).
- **Modular layout**: `app/core`, `app/api`, `app/services`; single responsibility per module.
- **Docstrings** on main functions and classes.

Skipped or minimal in this version: full async everywhere, token-usage logging, relevance filtering, auth, and multi-tenant indexing.

---

## g. How I used AI tools

- **Code generation**: Used an AI assistant to generate initial structure, I then adjusted types, error handling, and project layout to match my preferences.

---

## Limitations

- **Single-node Chroma**: Not distributed; fine for demo, not for high scale.
- **No auth**: Anyone with network access can upload and query.
- **Basic prompt-injection filter**: Deny-list is easy to bypass; production would need a stronger approach (e.g. classifier or dedicated guardrail service).
- **DOC/DOCX**: Relies on `unstructured` or `python-docx`; some edge formats may fail.
- **Tests** not performed fully. Some of the edge cases may fail.

---

## h. Future improvements

- **Auth and multi-tenancy**: Per-user or per-org collections and API keys.
- **Reranker**: Add a cross-encoder or reranker step after hybrid retrieval.
- **Token usage and cost**: Log and optionally expose token counts and cost per request.
- **Managed vector DB**: Migrate to OpenSearch, Pinecone, or pgvector for production scale.
- **LLM**: Best LLM (vLLMs) models hosted on GPU-enabled infrastructure like NVIDIA H100
- **AWS S3**: Store documents in Cloud platform like S3 buckets
- **Serverless**: Going with serverless approach for like Lambda
- **Docker**: Build, test, and deploy applications quickly
- **Monitoring**: Cloud monitoring for Performance and Scalability
- **Guardrails**: Scope to enhance safety, security, and ethical frameworks
- **Agentic**: User → Planner → Tools (vector search, web, calculator, DB, etc.) → LLM synthesizer → Answer. This approach helps if docs are outdates, incomplete or missing real information.
- **Memory Layer**: conversation memory
---

