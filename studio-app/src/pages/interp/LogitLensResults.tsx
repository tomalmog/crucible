import type { ReactNode } from "react";
import type { LogitLensResult, LogitLensTopToken } from "../../types/interp";
import { EvidenceSummary } from "./EvidenceSummary";
import { cleanToken, describeTopToken, formatPercent } from "./interpDisplay";

interface Props {
  result: LogitLensResult;
}

const CELL_W = 110;
const CELL_H = 28;
const LABEL_W = 80;
const HEADER_H = 32;
const PROB_LEGEND = [0.15, 0.35, 0.6, 0.85];

export function LogitLensResults({ result }: Props): ReactNode {
  const { input_tokens, layers } = result;
  const cols = input_tokens.length;
  const rows = layers.length;
  const svgW = LABEL_W + cols * CELL_W;
  const svgH = HEADER_H + rows * CELL_H;
  const topCell = findTopCell(result);

  const allUnknown = input_tokens.every(
    (t) => t === "<unk>" || t === "[UNK]",
  );

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Logit lens</span>
          <h3>Decoded token probabilities by layer</h3>
          <p>
            Each cell shows the top decoded token from the residual stream at
            one layer and input position. Intensity tracks probability.
          </p>
        </div>
        <span className="interp-evidence-badge">{cols} tokens · {rows} layers</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Input tokens", value: String(cols) },
          { label: "Layers decoded", value: String(rows) },
          { label: "Peak probability", value: topCell ? formatPercent(topCell.prob) : "n/a", tone: "positive" },
          { label: "Peak token", value: topCell ? cleanToken(topCell.token) : "n/a" },
        ]}
      />
      {allUnknown ? (
        <div className="info-banner">
          All input tokens are outside this model's vocabulary. This model was trained on
          domain-specific data with a small vocabulary — try using text from its training
          domain (e.g. "The transformer architecture uses self-attention").
        </div>
      ) : result.warning ? (
        <div className="info-banner">{result.warning}</div>
      ) : null}
      {!allUnknown && (
        <>
          <PredictionRibbon result={result} />
          <div className="interp-chart-frame chart-scroll-card">
            <ProbabilityLegend />
            <svg
              width={svgW}
              height={svgH}
              className="logit-lens-svg"
              role="img"
              aria-label="Logit lens token probability matrix"
            >
              {input_tokens.map((tok, col) => (
                <text
                  key={`h-${col}`}
                  x={LABEL_W + col * CELL_W + CELL_W / 2}
                  y={HEADER_H - 8}
                  textAnchor="middle"
                  className="chart-label"
                >
                  {cleanToken(tok)}
                </text>
              ))}

              {layers.map((layer, row) => (
                <g key={layer.layer_index} transform={`translate(0, ${HEADER_H + row * CELL_H})`}>
                  <text x={LABEL_W - 8} y={CELL_H / 2 + 4} textAnchor="end" className="chart-label">
                    L{layer.layer_index}
                  </text>
                  {layer.predictions.map((pred) => {
                    const top = pred.top_k[0];
                    if (!top) return null;
                    const opacity = Math.max(0.08, Math.min(1, top.prob));
                    return (
                      <g key={pred.token_position} transform={`translate(${LABEL_W + pred.token_position * CELL_W}, 0)`}>
                        <rect
                          className="logit-cell-rect"
                          width={CELL_W - 2}
                          height={CELL_H - 2}
                          rx={6}
                          fill="var(--accent)"
                          opacity={opacity}
                        />
                        <rect
                          width={CELL_W - 2}
                          height={CELL_H - 2}
                          rx={6}
                          fill="none"
                          stroke="var(--border)"
                          strokeWidth={0.5}
                        />
                        <text x={CELL_W / 2} y={CELL_H / 2 + 4} textAnchor="middle" className="chart-cell-text">
                          {cleanToken(top.token)} ({formatPercent(top.prob, 0)})
                        </text>
                        <title>
                          Layer {layer.layer_index}, token {pred.token_position}: {cleanToken(top.token)}
                          {" "}at {formatPercent(top.prob, 2)}
                        </title>
                      </g>
                    );
                  })}
                </g>
              ))}
            </svg>
          </div>
        </>
      )}
    </div>
  );
}

function PredictionRibbon({ result }: { result: LogitLensResult }): ReactNode {
  const firstLayer = result.layers[0];
  const finalLayer = result.layers[result.layers.length - 1];
  if (!firstLayer || !finalLayer) return null;

  return (
    <div className="prediction-ribbon" aria-label="First-to-final layer token predictions">
      {result.input_tokens.map((token, index) => {
        const early = firstLayer.predictions.find((p) => p.token_position === index)?.top_k[0];
        const final = finalLayer.predictions.find((p) => p.token_position === index)?.top_k[0];
        const didChange = early?.token !== final?.token;
        return (
          <article className="prediction-card" key={`${token}-${index}`}>
            <span className="prediction-token">{cleanToken(token)}</span>
            <div className="prediction-flow">
              <span>{describeTopToken(early)}</span>
              <i aria-hidden="true" />
              <strong className={didChange ? "changed" : ""}>{describeTopToken(final)}</strong>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ProbabilityLegend(): ReactNode {
  return (
    <div className="flex-row" aria-hidden="true">
      <span className="metric-label">Probability</span>
      {PROB_LEGEND.map((prob) => (
        <span className="flex-row-tight text-secondary text-sm" key={prob}>
          <span className={`probability-swatch probability-swatch-${Math.round(prob * 100)}`} />
          {(prob * 100).toFixed(0)}%
        </span>
      ))}
    </div>
  );
}

function findTopCell(result: LogitLensResult): LogitLensTopToken | null {
  let topCell: LogitLensTopToken | null = null;
  for (const layer of result.layers) {
    for (const prediction of layer.predictions) {
      const topToken = prediction.top_k[0];
      if (topToken && (!topCell || topToken.prob > topCell.prob)) {
        topCell = topToken;
      }
    }
  }
  return topCell;
}
