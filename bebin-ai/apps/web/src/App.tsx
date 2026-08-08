const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export function App() {
  return (
    <main className="app-shell">
      <section className="intro">
        <p className="eyebrow">Phase 6 Complete</p>
        <h1>Bebin AI</h1>
        <p className="summary">
          The platform now has a working FastAPI backend, React frontend, and
          deterministic ML foundation: dataset preprocessing plus a trainable
          BPE tokenizer feeding checkpointed Transformer training and local text generation.
        </p>
        <div className="status-grid">
          <div className="status-panel">
            <h2>Backend API</h2>
            <p>{apiBaseUrl}</p>
          </div>
          <div className="status-panel">
            <h2>Next</h2>
            <p>
              Phase 7 exposes the local model through FastAPI chat and streaming endpoints.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
