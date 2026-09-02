import { useState } from "react";
import { Search, Loader2 } from "lucide-react";

const API_URL = "http://localhost:8000/ask";

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error("Request failed");
      const data = await res.json();
      setResult(data);
    } catch {
      setError("Couldn't reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <h1>Diabetes Treatment Evidence</h1>
        <p className="subtitle">
          Ask a question about Type 2 diabetes drug therapy. Answers are
          synthesized from PubMed research and cited by source.
        </p>
      </header>

      <form className="search-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. How does metformin compare to GLP-1 agonists for weight loss?"
        />
        <button type="submit" disabled={loading}>
          {loading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className="answer">
            {result.answer.split("\n").map((line, i) => (
              <p key={i}>{line}</p>
            ))}
          </div>

          <div className="sources">
            <h2>Sources</h2>
            <ol>
              {result.sources.map((s, i) => (
                <li key={i}>
                  <span className="source-title">{s.title}</span>
                  <span className="source-meta">
                    PMID {s.pmid} · relevance {(1 - s.distance).toFixed(2)}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </div>
  );
}