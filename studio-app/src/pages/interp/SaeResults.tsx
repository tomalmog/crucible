import type { ReactNode } from "react";
import type { SaeTrainResult, SaeAnalyzeResult, SaeFeature } from "../../types/interp";
import { EvidenceSummary, type EvidenceMetric } from "./EvidenceSummary";
import { formatNumber, formatPercent } from "./interpDisplay";

const CW = 1000;
const CH = 380;
const B = { top: 50, right: 28, bottom: 68, left: 80 };

interface ChartDomain {
  min: number;
  max: number;
}

interface XTick {
  index: number;
  label: string;
}

export function SaeTrainResults({ result }: { result: SaeTrainResult }): ReactNode {
  const history = result.history ?? [];
  const firstLoss = history[0]?.loss;
  const lossChange = firstLoss ? ((firstLoss - result.final_loss) / firstLoss) * 100 : null;
  const expansionRatio = result.input_dim > 0 ? result.latent_dim / result.input_dim : 0;

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">SAE train</span>
          <h3>Sparse autoencoder loss curves</h3>
          <p>
            Train sparse features over activations to make candidate patterns
            easier to inspect.
          </p>
        </div>
        <span className="interp-evidence-badge">{result.layer_name}</span>
      </div>
      <EvidenceSummary
        items={buildSaeTrainMetrics(result, lossChange, expansionRatio)}
        label="Sparse autoencoder training summary"
      />
      <DictionaryDimensions result={result} expansionRatio={expansionRatio} />
      {history.length > 1 && <LossChart history={history} />}
      {result.sae_path && (
        <div className="interp-artifact-path">SAE saved to: {result.sae_path}</div>
      )}
    </div>
  );
}

function buildSaeTrainMetrics(
  result: SaeTrainResult,
  lossChange: number | null,
  expansionRatio: number,
): EvidenceMetric[] {
  const metrics: EvidenceMetric[] = [
    { label: "Final loss", value: formatNumber(result.final_loss, 4), tone: "positive" },
    { label: "Reconstruction", value: formatNumber(result.final_recon_loss, 4) },
    { label: "Sparsity", value: formatNumber(result.final_sparsity_loss, 4) },
    { label: "Loss reduction", value: lossChange !== null ? `${lossChange.toFixed(1)}%` : "n/a" },
    { label: "Dictionary", value: `${result.latent_dim}`, detail: `${expansionRatio.toFixed(1)}x expansion` },
  ];
  if (result.average_l0 !== undefined) {
    metrics.push({ label: "Average L0", value: formatNumber(result.average_l0, 1) });
  }
  if (result.dead_features !== undefined) {
    metrics.push({ label: "Dead features", value: String(result.dead_features), tone: "critical" });
  }
  if (result.fvu !== undefined) {
    metrics.push({ label: "FVU", value: formatNumber(result.fvu, 3) });
  }
  return metrics;
}

function DictionaryDimensions({
  expansionRatio,
  result,
}: {
  expansionRatio: number;
  result: SaeTrainResult;
}): ReactNode {
  return (
    <div className="dictionary-dimensions">
      <span className="metric-label">Dictionary geometry</span>
      <div>
        <strong>{result.input_dim}</strong>
        <span>activation dim</span>
      </div>
      <i aria-hidden="true" />
      <div>
        <strong>{result.latent_dim}</strong>
        <span>sparse features</span>
      </div>
      <small>{expansionRatio.toFixed(1)}x overcomplete</small>
    </div>
  );
}

interface EpochRow { epoch: number; loss: number; recon_loss: number; sparsity_loss: number }

const SERIES_DEFS: { key: keyof EpochRow; label: string; color: string }[] = [
  { key: "loss", label: "total loss", color: "var(--chart-1)" },
  { key: "recon_loss", label: "reconstruction", color: "var(--chart-2)" },
  { key: "sparsity_loss", label: "sparsity", color: "var(--chart-3)" },
];

function LossChart({ history }: { history: EpochRow[] }): ReactNode {
  const allY = history.flatMap((h) => [h.loss, h.recon_loss, h.sparsity_loss]);
  const yR = yDomain(allY);
  const yTicks = makeTicks(yR.min, yR.max, 5);
  const xTicks = xTickRows(history.map((h) => h.epoch), 6);
  const n = history.length;

  return (
    <div className="interp-chart-frame">
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

function linePath(vals: number[], yR: ChartDomain, n: number): string {
  return vals.map((v, i) => `${i === 0 ? "M" : "L"}${mX(i, n).toFixed(2)} ${mY(v, yR).toFixed(2)}`).join(" ");
}
function mX(i: number, n: number): number {
  const w = CW - B.left - B.right;
  return n <= 1 ? B.left + w / 2 : B.left + (i / (n - 1)) * w;
}
function mY(v: number, yR: ChartDomain): number {
  const h = CH - B.top - B.bottom;
  return CH - B.bottom - ((v - yR.min) / Math.max(yR.max - yR.min, 1e-6)) * h;
}
function yDomain(vals: number[]): ChartDomain {
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
function xTickRows(xVals: number[], count: number): XTick[] {
  if (xVals.length <= count) return xVals.map((v, i) => ({ index: i, label: String(v) }));
  const last = xVals.length - 1;
  return Array.from({ length: count }, (_, i) => {
    const idx = Math.round((i / (count - 1)) * last);
    return { index: idx, label: String(xVals[idx]) };
  });
}

export function SaeAnalyzeResults({ result }: { result: SaeAnalyzeResult }): ReactNode {
  const maxActivation = result.top_features.length > 0
    ? Math.max(...result.top_features.map((f) => f.activation), 0.01)
    : 1;
  const activePct = result.total_features > 0
    ? (result.active_features / result.total_features) * 100
    : 0;

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">SAE analyze</span>
          <h3>Activated sparse features</h3>
          <p>
            A dossier of the strongest latent features, tentative labels, and
            training examples they most resemble.
          </p>
        </div>
        <span className="interp-evidence-badge">{result.top_features.length} features</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Reconstruction error", value: formatNumber(result.reconstruction_error, 4) },
          { label: "Sparsity", value: formatPercent(result.sparsity), tone: "positive" },
          { label: "Active features", value: `${result.active_features} / ${result.total_features}` },
          { label: "Activation coverage", value: `${activePct.toFixed(1)}%` },
        ]}
        label="Sparse feature analysis summary"
      />
      <FeatureDensityRail activeFeatures={result.active_features} totalFeatures={result.total_features} />
      <div className="interp-callout">
        <span>Input analyzed</span>
        <p>{truncateText(result.input_text, 220)}</p>
      </div>
      {result.top_features.length > 0 && (
        <div className="feature-atlas-grid">
          {result.top_features.map((feature) => (
            <FeatureActivationRow
              feature={feature}
              key={feature.feature_index}
              maxActivation={maxActivation}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function FeatureDensityRail({
  activeFeatures,
  totalFeatures,
}: {
  activeFeatures: number;
  totalFeatures: number;
}): ReactNode {
  const activeShare = totalFeatures > 0 ? activeFeatures / totalFeatures : 0;
  return (
    <div className="feature-density-rail">
      <span className="metric-label">Sparse activation density</span>
      <progress value={activeShare} max={1}>{formatPercent(activeShare)}</progress>
      <strong>{formatPercent(activeShare, 2)} active</strong>
    </div>
  );
}

function FeatureActivationRow({
  feature,
  maxActivation,
}: {
  feature: SaeFeature;
  maxActivation: number;
}): ReactNode {
  const pct = Math.round((feature.activation / maxActivation) * 100);
  const concept = feature.concept && feature.concept !== "unknown" ? feature.concept : "Unlabeled";
  const textPreview = feature.associated_texts?.[0];

  return (
    <article className="feature-dossier-card">
      <div className="feature-dossier-topline">
        <span className="feature-index-pill">Feature #{feature.feature_index}</span>
        <strong>{formatNumber(feature.activation, 4)}</strong>
      </div>
      <h4>{concept}</h4>
      <progress className="feature-activation-meter" value={feature.activation} max={maxActivation}>
        {pct}%
      </progress>
      {textPreview && (
        <p>{truncateText(textPreview, 150)}</p>
      )}
    </article>
  );
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}
