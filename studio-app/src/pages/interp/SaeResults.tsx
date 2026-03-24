import type { SaeTrainResult, SaeAnalyzeResult } from "../../types/interp";

const CW = 1000;
const CH = 380;
const B = { top: 50, right: 28, bottom: 68, left: 80 };

export function SaeTrainResults({ result }: { result: SaeTrainResult }) {
  const history = result.history ?? [];

  return (
    <div className="panel stack-sm">
      <h3>SAE Training Results</h3>
      <div className="stats-grid">
        <div className="metric-card">
          <span className="metric-label">Final Loss</span>
          <span className="metric-value">{result.final_loss.toFixed(4)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Reconstruction Loss</span>
          <span className="metric-value">{result.final_recon_loss.toFixed(4)}</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">Sparsity Loss</span>
          <span className="metric-value">{result.final_sparsity_loss.toFixed(4)}</span>
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
      {history.length > 1 && <LossChart history={history} />}
      {result.sae_path && (
        <p className="text-secondary text-sm">SAE saved to: {result.sae_path}</p>
      )}
    </div>
  );
}

interface EpochRow { epoch: number; loss: number; recon_loss: number; sparsity_loss: number }

const SERIES_DEFS: { key: keyof EpochRow; label: string; color: string }[] = [
  { key: "loss", label: "total loss", color: "var(--chart-1)" },
  { key: "recon_loss", label: "reconstruction", color: "var(--chart-2)" },
  { key: "sparsity_loss", label: "sparsity", color: "var(--chart-3)" },
];

function LossChart({ history }: { history: EpochRow[] }) {
  const allY = history.flatMap((h) => [h.loss, h.recon_loss, h.sparsity_loss]);
  const yR = yDomain(allY);
  const yTicks = makeTicks(yR.min, yR.max, 5);
  const xTicks = xTickRows(history.map((h) => h.epoch), 6);
  const n = history.length;

  return (
    <div className="training-chart-card">
      <svg viewBox={`0 0 ${CW} ${CH}`} className="training-chart-svg">
        <text className="training-chart-title" x={CW / 2} y={26}>Training Loss</text>
        {yTicks.map((v) => (
          <g key={v.toFixed(6)}>
            <line className="training-grid-line" x1={B.left} x2={CW - B.right} y1={mY(v, yR)} y2={mY(v, yR)} />
            <text className="training-axis-tick training-axis-tick-y" x={B.left - 10} y={mY(v, yR) + 4}>{fmtY(v)}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={t.index}>
            <line className="training-grid-line vertical" x1={mX(t.index, n)} x2={mX(t.index, n)} y1={B.top} y2={CH - B.bottom} />
            <text className="training-axis-tick training-axis-tick-x" x={mX(t.index, n)} y={CH - B.bottom + 22}>{t.label}</text>
          </g>
        ))}
        <line className="training-axis-line" x1={B.left} x2={CW - B.right} y1={CH - B.bottom} y2={CH - B.bottom} />
        <line className="training-axis-line" x1={B.left} x2={B.left} y1={B.top} y2={CH - B.bottom} />
        {SERIES_DEFS.map((s) => (
          <path key={s.key} d={linePath(history.map((h) => Number(h[s.key])), yR, n)} fill="none" stroke={s.color} strokeWidth={2.5} />
        ))}
        <text className="training-axis-label" x={CW / 2} y={CH - 14}>Epoch</text>
        <text className="training-axis-label" x={20} y={CH / 2} transform={`rotate(-90 20 ${CH / 2})`}>Loss</text>
        <g transform={`translate(${CW - B.right - 280}, ${B.top - 24})`}>
          {SERIES_DEFS.map((s, i) => (
            <g key={s.key} transform={`translate(${i * 100}, 0)`}>
              <line x1={0} y1={9} x2={20} y2={9} stroke={s.color} strokeWidth={2.5} />
              <text className="training-legend-label" x={26} y={13}>{s.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

function linePath(vals: number[], yR: { min: number; max: number }, n: number): string {
  return vals.map((v, i) => `${i === 0 ? "M" : "L"}${mX(i, n).toFixed(2)} ${mY(v, yR).toFixed(2)}`).join(" ");
}
function mX(i: number, n: number): number {
  const w = CW - B.left - B.right;
  return n <= 1 ? B.left + w / 2 : B.left + (i / (n - 1)) * w;
}
function mY(v: number, yR: { min: number; max: number }): number {
  const h = CH - B.top - B.bottom;
  return CH - B.bottom - ((v - yR.min) / Math.max(yR.max - yR.min, 1e-6)) * h;
}
function yDomain(vals: number[]) {
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  if (Math.abs(max - min) < 1e-6) return { min: min - 0.5, max: max + 0.5 };
  const m = (max - min) * 0.08;
  return { min: Math.max(0, min - m), max: max + m };
}
function makeTicks(min: number, max: number, n: number): number[] {
  const step = (max - min) / (n - 1);
  return Array.from({ length: n }, (_, i) => min + i * step);
}
function fmtY(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(3);
}
function xTickRows(xVals: number[], count: number) {
  if (xVals.length <= count) return xVals.map((v, i) => ({ index: i, label: String(v) }));
  const last = xVals.length - 1;
  return Array.from({ length: count }, (_, i) => {
    const idx = Math.round((i / (count - 1)) * last);
    return { index: idx, label: String(xVals[idx]) };
  });
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
          <span className="metric-value">{result.reconstruction_error.toFixed(4)}</span>
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
            const concept = f.concept && f.concept !== "unknown" ? f.concept : null;
            return (
              <div key={f.feature_index} style={{ marginBottom: 6 }}>
                <div className="bar-row">
                  <div className="bar-label">
                    <span>#{f.feature_index}{concept ? ` \u2014 ${concept}` : ""}</span>
                    <strong>{f.activation.toFixed(4)}</strong>
                  </div>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
