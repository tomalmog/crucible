import type { ReactNode } from "react";
import type { PatchingLayerResult, PatchingResult } from "../../types/interp";
import { EvidenceSummary } from "./EvidenceSummary";
import { formatNumber, formatPercent } from "./interpDisplay";

interface Props {
  result: PatchingResult;
}

const CHART_W = 960;
const ROW_H = 34;
const LABEL_W = 160;
const VALUE_W = 90;

export function ActivationPatchingResults({ result }: Props): ReactNode {
  const { layer_results, clean_metric, corrupted_metric, metric } = result;
  const maxRecovery = Math.max(...layer_results.map((r) => Math.abs(r.recovery)), 0.01);
  const bestLayer = findBestLayer(result);
  const metricDelta = clean_metric - corrupted_metric;

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Activation patching</span>
          <h3>Recovery by layer</h3>
          <p>
            Clean activations are transplanted into the corrupted run layer by layer.
            Positive recovery marks layers where the patch moved the measured
            output toward the clean run.
          </p>
        </div>
        <span className="interp-evidence-badge">{metric}</span>
      </div>
      <PromptDuel cleanText={result.clean_text} corruptedText={result.corrupted_text} />
      <EvidenceSummary
        items={[
          { label: "Clean metric", value: formatNumber(clean_metric) },
          { label: "Corrupted metric", value: formatNumber(corrupted_metric), tone: "critical" },
          {
            detail: tokenPairLabel(result),
            label: "Metric delta",
            value: formatNumber(metricDelta),
          },
          {
            detail: bestLayer ? bestLayer.layer_name : undefined,
            label: "Strongest recovery",
            tone: "positive",
            value: bestLayer ? formatPercent(bestLayer.recovery) : "n/a",
          },
        ]}
      />
      {bestLayer && (
        <div className="interp-callout">
          <span>Strongest recovery layer</span>
          <strong>Layer {bestLayer.layer_index}</strong>
          <p>
            {bestLayer.layer_name} shifted the metric to {formatNumber(bestLayer.patched_metric)}
            {" "}with {formatPercent(bestLayer.recovery)} recovery.
          </p>
        </div>
      )}
      <CausalLayerStrip
        bestLayer={bestLayer}
        layers={layer_results}
        maxRecovery={maxRecovery}
      />
      {layer_results.length > 0 ? (
        <RecoveryChart maxRecovery={maxRecovery} result={result} />
      ) : (
        <p className="text-secondary text-sm">No layer recovery data returned.</p>
      )}
    </div>
  );
}

function tokenPairLabel(result: PatchingResult): string | undefined {
  if (!result.clean_token || !result.corrupt_token) return undefined;
  return `${result.clean_token} vs ${result.corrupt_token}`;
}

function PromptDuel({
  cleanText,
  corruptedText,
}: {
  cleanText: string;
  corruptedText: string;
}): ReactNode {
  return (
    <div className="prompt-duel">
      <article>
        <span>Clean prompt</span>
        <p>{cleanText}</p>
      </article>
      <article>
        <span>Corrupted prompt</span>
        <p>{corruptedText}</p>
      </article>
    </div>
  );
}

function CausalLayerStrip({
  bestLayer,
  layers,
  maxRecovery,
}: {
  bestLayer: PatchingLayerResult | null;
  layers: PatchingLayerResult[];
  maxRecovery: number;
}): ReactNode {
  if (layers.length === 0) return null;
  return (
    <div className="causal-layer-strip">
      <span className="metric-label">Layer recovery profile</span>
      <div className="causal-layer-cells">
        {layers.map((layer) => {
          const strength = Math.min(1, Math.abs(layer.recovery) / maxRecovery);
          const tone = layer.recovery >= 0 ? "positive" : "negative";
          const isBest = bestLayer?.layer_name === layer.layer_name;
          return (
            <span
              className={`causal-layer-cell ${tone}${isBest ? " best" : ""}`}
              key={layer.layer_name}
              style={{ opacity: 0.24 + strength * 0.76 }}
              title={`${layer.layer_name}: ${formatPercent(layer.recovery)}`}
            >
              L{layer.layer_index}
            </span>
          );
        })}
      </div>
    </div>
  );
}

function findBestLayer(result: PatchingResult): PatchingLayerResult | null {
  let bestLayer = result.layer_results[0] ?? null;
  for (const layer of result.layer_results) {
    if (!bestLayer || Math.abs(layer.recovery) > Math.abs(bestLayer.recovery)) {
      bestLayer = layer;
    }
  }
  return bestLayer;
}

function RecoveryChart({
  maxRecovery,
  result,
}: {
  maxRecovery: number;
  result: PatchingResult;
}): ReactNode {
  const zeroX = LABEL_W + (CHART_W - LABEL_W - VALUE_W) / 2;
  const barMaxW = (CHART_W - LABEL_W - VALUE_W) / 2 - 8;
  const chartH = result.layer_results.length * ROW_H + 24;

  return (
    <div className="interp-chart-frame chart-scroll-card">
      <div className="causal-map-title">
        <span>Layer recovery scan</span>
        <strong>negative</strong>
        <strong>positive</strong>
      </div>
      <svg width={CHART_W} height={chartH} className="training-chart-svg">
        <line
          className="training-axis-line"
          x1={zeroX}
          x2={zeroX}
          y1={8}
          y2={chartH - 8}
        />
        {result.layer_results.map((layer, index) => {
          const y = 18 + index * ROW_H;
          const width = (Math.abs(layer.recovery) / maxRecovery) * barMaxW;
          const isPositive = layer.recovery >= 0;
          const x = isPositive ? zeroX : zeroX - width;
          return (
            <g key={layer.layer_name}>
              <text className="training-axis-tick training-axis-tick-y" x={LABEL_W - 12} y={y + 5}>
                L{layer.layer_index}
              </text>
              <line
                className="training-grid-line"
                x1={LABEL_W}
                x2={CHART_W - VALUE_W}
                y1={y + 10}
                y2={y + 10}
              />
              <rect
                x={x}
                y={y - 8}
                width={width}
                height={16}
                rx={8}
                fill={isPositive ? "var(--success)" : "var(--error)"}
                opacity={0.85}
              />
              <text className="training-legend-label" x={CHART_W - VALUE_W + 12} y={y + 5}>
                {formatPercent(layer.recovery)}
              </text>
              <title>
                {layer.layer_name}: patched metric {formatNumber(layer.patched_metric, 4)}
              </title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
