# Expense Agent

An agent that takes a CSV of financial transactions and produces a structured
expense summary. It categorizes each transaction, computes totals per category,
flags unusual or high-value spends, and returns a concise report.

It runs two ways:

- **CLI** — analyze a CSV file and print or save the report.
- **Web app** — a React frontend with an upload button that sends the CSV to a
  FastAPI backend and prints the report below the button.

Categorization is dynamic: known expense types match fast offline keyword
rules, and any **new** kind of item is sent to an LLM (via Groq) that either
reuses an existing category or coins a new one. When the LLM is disabled or
unavailable, the agent falls back to offline rules only.

---

## How it works

A single-pass agentic loop chains four single-responsibility tools:

1. **File reader** — load transactions from CSV (or a plain-text list).
2. **Categorizer** — keyword rules first, LLM fallback for unknown items.
3. **Calculator** — totals per category and the grand total.
4. **Flagging** — rule-based detection of unusual / high-value spends
   (`HIGH_VALUE` = top ~5% by an adaptive percentile threshold; `UNUSUAL` =
   statistical z-score outlier).

The result is a structured report (transaction count, grand total, per-category
totals with shares and counts, and the flagged-transaction list).

---

## Technologies used

### Backend (Python)

| Technology | Purpose |
| --- | --- |
| **Python ≥ 3.13** | Language runtime |
| **uv** | Dependency management and script runner |
| **FastAPI** | HTTP API serving the `/analyze` endpoint |
| **Uvicorn** | ASGI server that runs the FastAPI app |
| **python-multipart** | Parses the uploaded CSV (multipart form data) |
| **LangChain + langchain-groq** | LLM client for dynamic categorization |
| **Groq** | LLM provider (default model `openai/gpt-oss-20b`) |
| **python-dotenv** | Loads API keys from `.env` |
| **pytest** | Test runner |

> The project also lists `langchain-community`, `langchain-huggingface`,
> `faiss-cpu`, `langgraph`, and `sentence-transformers` as dependencies. They
> are not required by the expense agent itself and can be trimmed if you don't
> use them elsewhere.

### Frontend (JavaScript)

| Technology | Purpose |
| --- | --- |
| **React 18** | UI components |
| **Vite 5** | Dev server, build tooling, and API proxy |
| **Node.js + npm** | Runtime and package manager |
| **Fetch API** | Uploads the CSV to the backend |

---

## Project structure

```
expance-calculator/
├── expense_calculator.py   # the agent: tools, pipeline, CLI
├── api.py                  # FastAPI backend (POST /analyze, GET /health)
├── pyproject.toml          # Python project + dependencies (uv)
├── sample_expenses.csv     # sample input data
├── tests/                  # pytest tests
└── frontend/               # React + Vite app
    ├── index.html
    ├── vite.config.js      # dev server + /api proxy to the backend
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx         # upload button + results tables
        └── index.css
```

---

## Setup

### Prerequisites

- Python 3.13+ and [uv](https://docs.astral.sh/uv/)
- Node.js 18+ and npm

### 1. Backend dependencies

From the project root:

```powershell
uv sync
```

### 2. Frontend dependencies

```powershell
cd frontend
npm install
```

### 3. API keys (for LLM categorization)

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Without a valid key, the agent still runs — unknown items just fall into the
`Other` category instead of getting a dynamically generated one.

> **Security:** `.env` is git-ignored. Never commit real API keys.

---

## Running the CLI

Analyze the sample CSV and print the report:

```powershell
uv run python expense_calculator.py
```

Analyze a specific file:

```powershell
uv run python expense_calculator.py path/to/transactions.csv
```

Useful flags:

```powershell
# Save structured JSON (any other extension saves the text summary)
uv run python expense_calculator.py -o report.json

# Skip the LLM and use offline keyword rules only (fast)
uv run python expense_calculator.py --no-llm

# Use a specific Groq model
uv run python expense_calculator.py --model openai/gpt-oss-20b
```

Expected input: a CSV with an expense-name column and an amount column (e.g.
`ExpenseName,Amount`). Common header aliases (`name`, `description`, `value`,
`cost`, etc.) are recognized automatically.

---

## Running the web app

The web app needs **two processes**: the backend API and the frontend dev
server. Run each in its own terminal.

**Terminal 1 — backend** (from the project root):

```powershell
uv run uvicorn api:app --reload --port 8000
```

**Terminal 2 — frontend** (from `frontend/`):

```powershell
cd frontend
npm run dev
```

Then open **http://localhost:5173**, choose a CSV, and click
**Upload & Analyze**. The categorized report appears below the button.

- Toggle **Use LLM for new categories** off for a faster, fully offline run.
- Vite proxies `/api/*` to the backend at `http://127.0.0.1:8000`, so no CORS
  configuration is needed in development.

### API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/analyze?use_llm=true` | Upload a CSV (form field `file`); returns the report JSON |
| `GET` | `/health` | Health check |

Example with curl:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze?use_llm=false" -F "file=@sample_expenses.csv"
```

---

## Running the tests

```powershell
uv run pytest
```

---

## Notes

- LLM categorization makes one call per **unique** item name (results are
  cached per run), so large files with many distinct names take longer. Use
  `--no-llm` or the UI toggle for a fast offline run.
- The high-value flagging threshold adapts to the dataset (95th percentile), so
  it surfaces the top spends regardless of the data's scale.
