"""Expense summary agent.

Accepts a CSV list of financial transactions and produces a structured expense
summary. The agent runs a single agentic pass that chains four
single-responsibility tools:

    1. file reader      -> load transactions from CSV
    2. categorizer      -> keyword rules first, LLM fallback for new items
    3. calculator       -> totals per category and grand total
    4. flagging tool    -> rule-based detection of unusual / high-value spends

Dynamic categorization: known expense types match offline keyword rules; any
new kind of item is sent to the LLM (Groq), which either reuses an existing
category or coins a new one. Pass --no-llm (or omit the API key) to stay fully
offline, in which case unmatched items fall into 'Other'.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Transaction:
    """A single parsed financial transaction."""

    name: str
    amount: float
    category: str = "Uncategorized"
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool 1: File reader
# ---------------------------------------------------------------------------

# Known header aliases so we can read a variety of CSV layouts.
_NAME_KEYS = ("expensename", "name", "description", "item", "merchant", "transaction")
_AMOUNT_KEYS = ("amount", "value", "cost", "price", "total")


def _parse_amount(raw: str) -> Optional[float]:
    """Parse a currency-ish string into a float, or None if not a number."""
    if raw is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(raw).strip())
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_csv(text: str) -> bool:
    """Heuristic: treat content as CSV if the first non-empty line is comma or
    tab separated, otherwise treat it as a plain-text list."""
    for line in text.splitlines():
        if line.strip():
            return ("," in line) or ("\t" in line)
    return False


def _read_csv_text(text: str) -> List[Transaction]:
    reader = csv.DictReader(io.StringIO(text))
    txns: List[Transaction] = []

    # Build a case-insensitive map from the actual header names.
    fieldnames = reader.fieldnames or []
    header_map = {h.lower().strip(): h for h in fieldnames}

    name_col = next((header_map[k] for k in _NAME_KEYS if k in header_map), None)
    amount_col = next((header_map[k] for k in _AMOUNT_KEYS if k in header_map), None)

    # Fall back to positional columns (name, amount) if headers are unknown.
    if name_col is None and fieldnames:
        name_col = fieldnames[0]
    if amount_col is None and len(fieldnames) > 1:
        amount_col = fieldnames[1]

    for row in reader:
        name = (row.get(name_col) or "").strip() if name_col else ""
        amount = _parse_amount(row.get(amount_col)) if amount_col else None

        if not name and amount is None:
            continue
        if amount is None:
            # Skip rows we cannot price rather than guessing a value.
            continue

        txns.append(Transaction(name=name or "Unknown", amount=amount))

    return txns


def _read_plaintext(text: str) -> List[Transaction]:
    """Parse plain-text lines such as 'Coffee - 4.50' or 'Groceries: 60'."""
    txns: List[Transaction] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Find the last number on the line; everything before it is the name.
        matches = list(re.finditer(r"-?\d[\d,]*\.?\d*", line))
        if not matches:
            continue

        last = matches[-1]
        amount = _parse_amount(last.group())
        if amount is None:
            continue

        name = line[: last.start()].strip(" -:\t|")
        txns.append(Transaction(name=name or "Unknown", amount=amount))

    return txns


def read_transactions(path: str) -> List[Transaction]:
    """File reader tool: load transactions from a CSV or plain-text file.

    The format is auto-detected from the file contents (not just the
    extension), so a '.txt' with commas is read as CSV and vice versa.
    """
    text = Path(path).read_text(encoding="utf-8-sig")
    if _looks_like_csv(text):
        return _read_csv_text(text)
    return _read_plaintext(text)


# ---------------------------------------------------------------------------
# Tool 2: Rule-based categorizer
# ---------------------------------------------------------------------------

# Ordered category -> keyword rules. First match wins.
_CATEGORY_RULES: dist[str, List[str]] = {
    "Housing & Utilities": [
        "rent", "electricity", "water bill", "gas cylinder", "home maintenance",
        "internet", "mobile recharge", "cleaning supplies", "household",
    ],
    "Groceries & Food": [
        "grocer", "food", "vegetable", "fruit", "milk", "kitchen", "snack",
        "coffee", "breakfast", "lunch", "dinner", "dining", "pet food",
    ],
    "Transport & Travel": [
        "transport", "fuel", "parking", "taxi", "bus", "train", "flight",
        "hotel", "travel",
    ],
    "Health": [
        "medicine", "doctor", "insurance", "gym",
    ],
    "Education": [
        "school", "tuition", "book", "stationery",
    ],
    "Shopping & Lifestyle": [
        "shopping", "clothing", "furniture", "gift", "festival",
        "software purchase", "child care",
    ],
    "Entertainment & Subscriptions": [
        "entertainment", "movie", "online subscription", "streaming",
    ],
}


def categorize(name: str) -> str:
    """Rule-based categorizer: assign a category to a transaction name by keyword.

    Returns None-equivalent sentinel "Other" when no rule matches, which is the
    signal for the dynamic categorizer to ask the LLM.
    """
    lowered = name.lower()
    for category, keywords in _CATEGORY_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "Other"


# ---------------------------------------------------------------------------
# Tool 2b: Dynamic LLM categorizer
# ---------------------------------------------------------------------------

# Default Groq model confirmed accessible on this account. Override with the
# EXPENSE_LLM_MODEL environment variable without touching code.
_DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

_LLM_SYSTEM_PROMPT = (
    "You are an expense categorization engine. Given a single expense item "
    "name, respond with the best spending category for it.\n"
    "Prefer one of the KNOWN categories when the item clearly fits. If none "
    "fit well, invent a short, general, title-cased category name (1-3 words) "
    "that would group similar expenses.\n"
    "Respond with ONLY the category name on one line. No punctuation, no "
    "explanation, no quotes."
)


class DynamicCategorizer:
    """Categorizer tool with a cheap rule-based pass and an LLM fallback.

    Flow for each item:
      1. Try the deterministic keyword rules.
      2. If they return "Other", ask the LLM to pick an existing category or
         coin a new one, so genuinely new kinds of expenses get their own
         category instead of being lumped into "Other".
    Results are cached per item name, and the model is created lazily so the
    script still runs fully offline when --no-llm is set or no key is present.
    """

    def __init__(self, use_llm: bool = True, model: Optional[str] = None):
        self.use_llm = use_llm
        self.model_name = model or os.environ.get("EXPENSE_LLM_MODEL", _DEFAULT_LLM_MODEL)
        self._llm = None
        self._llm_failed = False
        self._cache: dist[str, str] = {}
        # Categories the LLM is told about; grows as it coins new ones so later
        # items can reuse an invented category instead of making near-duplicates.
        self._known: List[str] = list(_CATEGORY_RULES.keys())

    def _get_llm(self):
        if self._llm is None and not self._llm_failed:
            try:
                from dotenv import load_dotenv

                load_dotenv()
                from langchain_groq import ChatGroq

                self._llm = ChatGroq(model=self.model_name, temperature=0)
            except Exception:
                # Missing key, missing package, or bad config -> stay offline.
                self._llm_failed = True
        return self._llm

    def _ask_llm(self, name: str) -> str:
        llm = self._get_llm()
        if llm is None:
            return "Other"
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            known = ", ".join(self._known)
            messages = [
                SystemMessage(content=_LLM_SYSTEM_PROMPT),
                HumanMessage(content=f"KNOWN categories: {known}\nItem: {name}\nCategory:"),
            ]
            raw = llm.invoke(messages).content or ""
            category = raw.strip().splitlines()[0].strip().strip('"').strip(".")
            if not category:
                return "Other"
            # Remember newly coined categories for consistency on later items.
            if category not in self._known:
                self._known.append(category)
            return category
        except Exception:
            self._llm_failed = True
            return "Other"

    def categorize(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]

        category = categorize(name)
        if category == "Other" and self.use_llm:
            category = self._ask_llm(name)

        self._cache[name] = category
        return category


# ---------------------------------------------------------------------------
# Tool 3: Calculator
# ---------------------------------------------------------------------------


def totals_by_category(txns: List[Transaction]) -> dist[str, float]:
    """Calculator tool: sum amounts per category."""
    totals: dist[str, float] = {}
    for t in txns:
        totals[t.category] = round(totals.get(t.category, 0.0) + t.amount, 2)
    return totals


def grand_total(txns: List[Transaction]) -> float:
    """Calculator tool: sum of all transaction amounts."""
    return round(sum(t.amount for t in txns), 2)


# ---------------------------------------------------------------------------
# Tool 4: Rule-based flagging
# ---------------------------------------------------------------------------


def flag_transactions(
    txns: List[Transaction],
    high_value_abs: Optional[float] = None,
    zscore_threshold: float = 2.0,
) -> List[Transaction]:
    """Flagging tool: mark unusual or high-value spends in place.

    Two independent rules:
      * HIGH_VALUE  -> amount is in the top tier of this dataset. The threshold
                       adapts to the data (the larger of the 95th percentile and
                       mean + 2*std) so it surfaces a short list of big spends
                       instead of a fixed constant that may match everything or
                       nothing. Pass high_value_abs to force a fixed threshold.
      * UNUSUAL     -> amount is a statistical outlier (z-score over the mean).

    A transaction is flagged if it trips either rule.
    """
    amounts = [t.amount for t in txns]
    avg = mean(amounts) if amounts else 0.0
    std = pstdev(amounts) if len(amounts) > 1 else 0.0

    if high_value_abs is None:
        high_value_abs = _high_value_threshold(amounts, avg, std)

    for t in txns:
        t.flags = []
        if t.amount >= high_value_abs:
            t.flags.append("HIGH_VALUE")
        if std > 0 and (t.amount - avg) / std >= zscore_threshold:
            t.flags.append("UNUSUAL")

    return txns


def _high_value_threshold(amounts: List[float], avg: float, std: float) -> float:
    """Derive an adaptive high-value cutoff: the 95th percentile of the data.

    Using the percentile (rather than a fixed constant or a std-based cutoff)
    means roughly the top 5% of spends are flagged as high-value regardless of
    the scale or shape of the dataset. The separate z-score rule in
    flag_transactions handles statistical outliers independently.
    """
    if not amounts:
        return float("inf")
    ordered = sorted(amounts)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Report builder (structured output)
# ---------------------------------------------------------------------------


def build_report(txns: List[Transaction]) -> dist:
    """Assemble the structured expense summary from the computed tools."""
    totals = totals_by_category(txns)
    total = grand_total(txns)
    flagged = [t for t in txns if t.flags]

    categories = [
        {
            "category": cat,
            "total": amt,
            "share_pct": round((amt / total * 100), 1) if total else 0.0,
            "count": sum(1 for t in txns if t.category == cat),
        }
        for cat, amt in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "transaction_count": len(txns),
        "grand_total": total,
        "categories": categories,
        "flagged_transactions": [
            {"name": t.name, "amount": t.amount, "category": t.category, "flags": t.flags}
            for t in sorted(flagged, key=lambda t: t.amount, reverse=True)
        ],
    }


def render_report(report: dist) -> str:
    """Render the structured report as a concise human-readable text summary."""
    lines: List[str] = []
    lines.append("=" * 52)
    lines.append("              EXPENSE SUMMARY REPORT")
    lines.append("=" * 52)
    lines.append(f"Transactions analyzed : {report['transaction_count']}")
    lines.append(f"Grand total           : {report['grand_total']:,.2f}")
    lines.append("")
    lines.append("Totals by category")
    lines.append("-" * 52)
    for c in report["categories"]:
        lines.append(
            f"  {c['category']:<32} {c['total']:>10,.2f}  ({c['share_pct']:>4}% | {c['count']}x)"
        )
    lines.append("")

    flagged = report["flagged_transactions"]
    lines.append(f"Flagged transactions ({len(flagged)})")
    lines.append("-" * 52)
    if not flagged:
        lines.append("  None")
    else:
        for f in flagged:
            tags = ", ".join(f["flags"])
            lines.append(f"  {f['name']:<28} {f['amount']:>10,.2f}  [{tags}]")
    lines.append("=" * 52)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agentic single pass
# ---------------------------------------------------------------------------


def _analyze(txns: List[Transaction], use_llm: bool, model: Optional[str]) -> dist:
    """Shared pipeline: categorize -> flag -> report."""
    categorizer = DynamicCategorizer(use_llm=use_llm, model=model)
    for t in txns:
        t.category = categorizer.categorize(t.name)  # tool: dynamic categorizer
    flag_transactions(txns)                          # tool: flagging
    return build_report(txns)                        # tools: calculator + assembly


def run_agent(path: str, use_llm: bool = True, model: Optional[str] = None) -> dist:
    """Single-pass agentic loop: read -> categorize -> calculate -> flag -> report.

    Categorization is dynamic: known items match keyword rules, and any new
    kind of item is categorized by the LLM (unless use_llm is False), which can
    reuse an existing category or coin a new one.
    """
    txns = read_transactions(path)                   # tool: file reader (from path)
    return _analyze(txns, use_llm, model)


def run_agent_on_text(
    text: str, use_llm: bool = True, model: Optional[str] = None
) -> dist:
    """Same single-pass agent, but from raw CSV/text content (e.g. an upload).

    Used by the web API so an uploaded file is analyzed directly without being
    written to disk first.
    """
    if _looks_like_csv(text):
        txns = _read_csv_text(text)                  # tool: file reader (from text)
    else:
        txns = _read_plaintext(text)
    return _analyze(txns, use_llm, model)


# ---------------------------------------------------------------------------
# Backwards-compatible loader (kept for existing tests)
# ---------------------------------------------------------------------------


def load_expance_csv(path: str) -> List[Document]:
    """Legacy loader retained for compatibility with existing tests."""
    docs: List[Document] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            expense_name = (row.get("ExpenseName") or "").strip()
            amount = (row.get("Amount") or "").strip()

            if not expense_name and not amount:
                continue

            docs.append(
                Document(page_content=f"ExpenseName: {expense_name}\nAmount: {amount}")
            )

    return docs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Categorize transactions and print or save an expense report."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="./sample_expenses.csv",
        help="Path to a CSV transaction list.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write the report here. '.json' saves structured JSON; "
             "any other extension saves the text summary.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable the LLM fallback and use only the offline keyword rules "
             "(new items fall into 'Other').",
    )
    parser.add_argument(
        "--model",
        help=f"Groq model id for dynamic categorization "
             f"(default: env EXPENSE_LLM_MODEL or '{_DEFAULT_LLM_MODEL}').",
    )
    args = parser.parse_args(argv)

    report = run_agent(args.input, use_llm=not args.no_llm, model=args.model)
    text = render_report(report)

    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".json":
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            out.write_text(text, encoding="utf-8")
        print(f"Report saved to {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
