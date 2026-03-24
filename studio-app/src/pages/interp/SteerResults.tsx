import type { SteerComputeResult, SteerApplyResult } from "../../types/interp";

export function SteerComputeResults({ result }: { result: SteerComputeResult }) {
  return (
    <div className="panel stack-sm">
      <h3>Steering Vector Computed</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Vector Norm</span>
          <span className="metric-value">{result.vector_norm.toFixed(4)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Cosine Similarity</span>
          <span className="metric-value">{result.cosine_similarity.toFixed(4)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Positive Samples</span>
          <span className="metric-value">{result.num_positive}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Negative Samples</span>
          <span className="metric-value">{result.num_negative}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Layer</span>
          <span className="metric-value text-sm">{result.layer_name}</span>
        </div>
      </div>
      {result.steering_vector_path && (
        <p className="text-secondary text-sm">Vector saved to: {result.steering_vector_path}</p>
      )}
    </div>
  );
}

export function SteerApplyResults({ result }: { result: SteerApplyResult }) {
  return (
    <div className="panel stack-sm">
      <h3>Steered Generation</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Coefficient</span>
          <span className="metric-value">{result.coefficient}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Layer</span>
          <span className="metric-value text-sm">{result.layer_name}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Max New Tokens</span>
          <span className="metric-value">{result.max_new_tokens}</span>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <h4 style={{ marginBottom: 4 }}>Original</h4>
          <pre className="console" style={{ fontSize: "0.8125rem", whiteSpace: "pre-wrap" }}>
            <strong>{result.input_text}</strong>{"\n\n"}{result.original_text}
          </pre>
        </div>
        <div>
          <h4 style={{ marginBottom: 4 }}>Steered</h4>
          <pre className="console" style={{ fontSize: "0.8125rem", whiteSpace: "pre-wrap" }}>
            <strong>{result.input_text}</strong>{"\n\n"}{result.steered_text}
          </pre>
        </div>
      </div>
    </div>
  );
}
