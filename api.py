"""FastAPI backend for the expense agent.

Exposes the existing expense agent over HTTP so a frontend can upload a CSV
instead of relying on the static sample file. A single endpoint accepts an
uploaded CSV, runs the agent on its contents, and returns the structured
report as JSON.

Run with:
    uv run uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from expense_calculator import run_agent_on_text

app = FastAPI(title="Expense Agent API", version="1.0.0")

# Allow the Vite/React dev server (and any local origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    use_llm: bool = Query(
        True,
        description="Use the LLM to categorize unknown items. "
        "Set false for a faster, fully offline run.",
    ),
) -> dict:
    """Analyze an uploaded CSV of transactions and return the expense report."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            raise HTTPException(
                status_code=400, detail="Could not decode file as text."
            )

    try:
        report = run_agent_on_text(text, use_llm=use_llm)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    if report["transaction_count"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid transactions found. Expected columns like "
            "'ExpenseName' and 'Amount'.",
        )

    report["filename"] = file.filename
    return report
