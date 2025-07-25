---

# 🧠 Smart Assistant – Document + Structured Data Chatbot

An AI-powered assistant that allows users to **chat with documents** and **structured datasets** through a conversational UI. It performs **semantic search**, **section-wise summarization**, and **structured data lookup** — intelligently switching tools based on user queries and data availability.

---

## 🔧 Features

### 📄 Document Understanding

* Upload and index documents (`.pdf`, `.docx`, `.txt`)
* Ask questions about content (retrieves facts using reranked document chunks)
* Summarize specific sections, generate insights, extract key points

### 📊 Structured Data Interaction

* Accesses an in-memory DataFrame when document data is insufficient
* Automatically escalates to tabular querying for numeric, comparative, or summary-based questions

### 🧠 Tool Orchestration via LangGraph

* Dynamically selects tools:

  * `document_search` → factual queries
  * `summarization_tool` → high-level overviews
  * `structured_data_lookup` → fallback for data not found in documents
* Reranking via `CrossEncoder` improves search relevance

### 🗂️ Conversational Memory

* Maintains context across chat turns via `LangGraph` memory
* Understands follow-ups like “What more about Product A?”

### 🪄 Transparent Reasoning

Every response includes:

* Final Answer
* Tool Trace
* Reasoning
* Source of truth (document or structured)

---

## 🚀 Getting Started

### 1. 📦 Install Dependencies

```bash
uv sync  # or pip install -r requirements.txt if using pip
```

---

### 2. ▶️ Launch the App

```bash
python -m src.app
```

---

### 3. 🧪 Try Queries Like:

| Goal                               | Sample Queries                                                | Tool Path                                     |
| ---------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| 📘 Ask for specific data from docs | *"What’s the North region revenue for Product C?"*            | `document_search`                             |
| 📉 Escalate to data lookup         | *"What’s the profit margin of Product A across all regions?"* | `document_search → structured_data_lookup`    |
| 🧾 Summarize section               | *"Give a high-level summary of Q4 performance"*               | `summarization_tool`                          |
| 📊 Summary fails → table           | *"Compare online vs offline sales by region"*                 | `summarization_tool → structured_data_lookup` |

---

## 📁 Project Structure

```bash
.
├── app.py               # Gradio UI
├── graph.py             # LangGraph logic with assistant orchestration
├── indexer.py           # Loads and indexes PDF, DOCX, TXT with embeddings
├── tools.py             # Tool definitions: search, summarize, structured query
├── logger.py            # Logging + callback tracing
├── pyproject.toml       # Python project configuration
├── uv.lock              # Lockfile for uv + reproducible builds
└── vectorstore/index/   # FAISS vector index (generated at runtime)
```

---

## 📚 Data Used

* 📄 **Document Corpus**: Any uploaded `.pdf`, `.txt`, or `.docx`
* 📊 **Structured DataFrame**: Pre-loaded in-memory dummy data, with:

  ```python
  columns = ["product", "region", "revenue", "online_pct", "profit_margin", "growth_qoq", "avg_order_value"]
  ```

---

## 🧠 Behind the Scenes

* **Semantic Search** via FAISS + `sentence-transformers`
* **Relevance Reranking** via `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **LLM**: `gpt-4o-mini` via `langchain-openai`
* **Prompt Reflection**: Decides whether to escalate based on LLM's self-check

---


Absolutely — here's the updated `README.md` section including all `.env` assumptions used in your code:

---

## 🔐 Environment Variables (`.env`)

To configure the assistant securely, you need to define the following environment variables in a `.env` file (at the root of the project)(`src`)):

```ini
# .env

OPENAI_API_KEY=your_openai_key_here
```

### 🔍 Notes:

* `OPENAI_API_KEY` is used for invoking the LLM (`ChatOpenAI`).
* You can configure the model used (optional)


> The environment variables are loaded using `python-dotenv` and automatically picked up when running `app.py`.

---

## 🧠 TODO

* **Refactoring of Prompt into separate module** 
* **Refactoring of tools into subpackage with module for each** 
---