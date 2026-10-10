from pathlib import Path

from expense_calculator import load_expance_csv


def test_load_expance_csv_reads_rows():
    csv_path = Path(__file__).resolve().parents[1] / "sample_expenses.csv"

    docs = load_expance_csv(str(csv_path))

    # Don't hardcode the row count or amounts: the sample data file may be
    # regenerated. Assert the loader returns rows in the expected shape.
    assert len(docs) > 0
    assert docs[0].page_content.startswith("ExpenseName: ")
    assert "Amount: " in docs[0].page_content
