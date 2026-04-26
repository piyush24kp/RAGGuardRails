# RAG with RBAC, Guardrails & Monitoring

An internal company chatbot built with Retrieval-Augmented Generation (RAG) that enforces **Role-Based Access Control (RBAC)**, protects sensitive data with **guardrails**, and is designed for production monitoring with **LangSmith** and **Ragas**.


---

## What it does

Employees of FinSolve Technologies ask questions in natural language. The system retrieves answers **only from documents the user's role is authorized to see** — HR staff can't read financial reports, finance staff can't read HR records, and only C-level executives have full access.

| Role | Accessible departments |
|---|---|
| HR | `hr`, `general` |
| Finance | `finance`, `general` |
| Marketing | `marketing`, `general` |
| Engineering | `engineering`, `general` |
| CEO | All departments |

---

## Architecture

```
User Query
    │
    ├─► Greeting check (regex, zero LLM cost)
    │       └─► Direct reply
    │
    ├─► Scope guardrail (LLM classifier)
    │       └─► Reject off-topic questions
    │
    ├─► RBAC-filtered retrieval (ChromaDB metadata filter)
    │       └─► Only chunks from allowed departments
    │
    ├─► Groq LLM (Llama 3.3 70B) → answer grounded in context
    │
    └─► PII guardrail (regex redaction on output)
            └─► Redact emails, phones, salaries, DOBs before display
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via [Groq Cloud](https://console.groq.com/) (free) |
| Embeddings | `all-MiniLM-L6-v2` (local, no API key needed) |
| Vector DB | ChromaDB (persistent, local) |
| Framework | LangChain + Python |
| Frontend | Streamlit |
| Monitoring | LangSmith |
| Evaluation | Ragas (faithfulness, answer relevancy, context precision) |

---

## Project Structure

```
RAGGuardRails/
├── DS-RPC-01/
│   └── data/                  # Company documents organized by department
│       ├── finance/           # Financial reports (finance + CEO access)
│       ├── hr/                # Employee data & handbook (HR + CEO access)
│       ├── marketing/         # Marketing reports (marketing + CEO access)
│       ├── engineering/       # Engineering docs (engineering + CEO access)
│       └── general/           # Policies accessible to all roles
├── ingestion/
│   └── loader.py              # Doc loading, chunking, incremental vector store updates
├── retrieval/
│   ├── rbac.py                # Role → department access mapping
│   ├── vector_store.py        # ChromaDB retriever with RBAC metadata filter
│   └── rag_chain.py           # RAG chain: retrieval + Groq LLM
├── guardrails/
│   ├── scope_check.py         # Out-of-scope question detection (LLM classifier)
│   └── pii_filter.py          # PII redaction on LLM output (regex)
├── app/
│   └── main.py                # Streamlit chat UI
├── evaluation/
│   └── ragas_eval.py          # Ragas evaluation runner
├── config.py                  # Central config loaded from .env
└── requirements.txt
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/piyush24kp/RAGGuardRails.git
cd RAGGuardRails
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
GROQ_API_KEY=your_key_here          # https://console.groq.com/ (free)
LANGCHAIN_API_KEY=your_key_here     # https://smith.langchain.com/ (free)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-rbac-chatbot
```

### 3. Build the vector store

Runs once. Re-run whenever documents change — only changed files are re-embedded.

```bash
python -m ingestion.loader
```

To force a full rebuild:

```bash
python -m ingestion.loader --full
```

### 4. Launch the app

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501), select a demo user from the sidebar, and start chatting.

---

## Demo users

| Username | Role | Can access |
|---|---|---|
| alice | HR | HR docs, general policies |
| bob | Finance | Financial reports, general policies |
| carol | Marketing | Marketing reports, general policies |
| dave | Engineering | Engineering docs, general policies |
| eve | CEO | Everything |

---

## Guardrails

**Input — Scope check**
Questions unrelated to company business (e.g. "Write me a poem") are rejected before reaching the retrieval pipeline.

**Input — Greeting detection**
Greetings ("Hi", "Hello", etc.) get a direct response with zero LLM/retrieval cost.

**Output — PII redaction**
The LLM response is scanned before display. Emails, phone numbers, salaries, dates of birth, and employee IDs are replaced with `[REDACTED]` tags.

---

## Evaluation

Run Ragas metrics on the built-in test set:

```bash
python -m evaluation.ragas_eval
```

Metrics reported: `faithfulness`, `answer_relevancy`, `context_precision`.

---

## Key design decisions

**Incremental ingestion** — A SHA-256 hash is stored per file. On re-run, only changed or new files are re-embedded and old chunks for that file are deleted first. Unchanged files are skipped entirely.

**Chunk size: 800 tokens, overlap: 100** — Chosen for the narrative structure of the markdown reports. Increase overlap or decrease chunk size if `context_precision` drops in Ragas evals.

**Embedding model: `all-MiniLM-L6-v2`** — Fast, runs locally with no API key, 384-dimensional vectors. Upgrade to `all-mpnet-base-v2` (768d) if retrieval quality is insufficient.

**ChromaDB `hnsw:space: cosine`** — Cosine similarity measures the angle between embedding vectors, which captures semantic direction better than Euclidean distance for text.

**Classifier temperature: 0.0** — The scope classifier must return exactly `IN_SCOPE` or `OUT_OF_SCOPE`. Any temperature above 0 risks non-deterministic output that breaks the string match.
