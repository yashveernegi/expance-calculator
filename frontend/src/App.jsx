import { useState } from "react";

function formatMoney(n) {
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export default function App() {
  const [file, setFile] = useState(null);
  const [useLlm, setUseLlm] = useState(true);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAnalyze() {
    if (!file) {
      setError("Please choose a CSV file first.");
      return;
    }
    setError("");
    setReport(null);
    setLoading(true);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`/api/analyze?use_llm=${useLlm}`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }

      setReport(await res.json());
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h1>Expense Agent</h1>
      <p className="subtitle">
        Upload a CSV of transactions to get a categorized expense report.
      </p>

      <div className="controls">
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <label className="llm-toggle">
          <input
            type="checkbox"
            checked={useLlm}
            onChange={(e) => setUseLlm(e.target.checked)}
          />
          Use LLM for new categories (slower)
        </label>
        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Upload & Analyze"}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {report && (
        <div className="results">
          <div className="summary">
            <span>
              <strong>{report.transaction_count}</strong> transactions
            </span>
            <span>
              Grand total: <strong>{formatMoney(report.grand_total)}</strong>
            </span>
            {report.filename && <span className="fname">{report.filename}</span>}
          </div>

          <h2>Totals by category</h2>
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th className="num">Total</th>
                <th className="num">Share</th>
                <th className="num">Count</th>
              </tr>
            </thead>
            <tbody>
              {report.categories.map((c) => (
                <tr key={c.category}>
                  <td>{c.category}</td>
                  <td className="num">{formatMoney(c.total)}</td>
                  <td className="num">{c.share_pct}%</td>
                  <td className="num">{c.count}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2>Flagged transactions ({report.flagged_transactions.length})</h2>
          {report.flagged_transactions.length === 0 ? (
            <p className="muted">None</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Category</th>
                  <th className="num">Amount</th>
                  <th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {report.flagged_transactions.map((f, i) => (
                  <tr key={`${f.name}-${i}`}>
                    <td>{f.name}</td>
                    <td>{f.category}</td>
                    <td className="num">{formatMoney(f.amount)}</td>
                    <td>
                      {f.flags.map((flag) => (
                        <span key={flag} className="tag">
                          {flag}
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
