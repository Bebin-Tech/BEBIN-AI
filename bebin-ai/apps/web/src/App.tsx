const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export function App() {
  return (
    <main className="app-shell">
      <section className="intro">
        <p className="eyebrow">Phase 1</p>
        <h1>Bebin AI</h1>
        <p className="summary">
          The monorepo is ready for the real assistant build: FastAPI on the
          backend, React on the frontend, and a future SLM training stack that
          stays fully owned by this project.
        </p>
        <div className="status-grid">
          <div className="status-panel">
            <h2>API</h2>
            <p>{apiBaseUrl}</p>
          </div>
          <div className="status-panel">
            <h2>Next</h2>
            <p>
              Phase 2 adds the dataset ingestion and validation pipeline.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}
