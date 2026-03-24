import type { SaeTrainResult, SaeAnalyzeResult } from "../../types/interp";

export function SaeTrainResults({ result }: { result: SaeTrainResult }) {
  return (
    <div className="panel stack-sm">
      <h3>SAE Training Results</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Final Loss</span>
          <span className="metric-value">{result.final_loss.toFixed(6)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Reconstruction Loss</span>
          <span className="metric-value">{result.final_recon_loss.toFixed(6)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Sparsity Loss</span>
          <span className="metric-value">{result.final_sparsity_loss.toFixed(6)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Epochs</span>
          <span className="metric-value">{result.epochs}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Input Dim</span>
          <span className="metric-value">{result.input_dim}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Latent Dim</span>
          <span className="metric-value">{result.latent_dim}</span>
        </div>
      </div>
      {result.sae_path && (
        <p className="text-secondary text-sm">SAE saved to: {result.sae_path}</p>
      )}
    </div>
  );
}

export function SaeAnalyzeResults({ result }: { result: SaeAnalyzeResult }) {
  const maxActivation = result.top_features.length > 0
    ? Math.max(...result.top_features.map((f) => f.activation), 0.01)
    : 1;

  return (
    <div className="panel stack-sm">
      <h3>SAE Analysis Results</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Reconstruction Error</span>
          <span className="metric-value">{result.reconstruction_error.toFixed(6)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Sparsity</span>
          <span className="metric-value">{(result.sparsity * 100).toFixed(1)}%</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Active Features</span>
          <span className="metric-value">{result.active_features} / {result.total_features}</span>
        </div>
      </div>
      {result.top_features.length > 0 && (
        <div>
          <h4 style={{ marginBottom: 8 }}>Top Features by Activation</h4>
          {result.top_features.map((f) => {
            const pct = Math.round((f.activation / maxActivation) * 100);
            return (
              <div className="bar-row" key={f.feature_index}>
                <div className="bar-label">
                  <span>#{f.feature_index}</span>
                  <strong>{f.activation.toFixed(4)}</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
