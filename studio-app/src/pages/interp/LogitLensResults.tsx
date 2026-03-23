import type { LogitLensResult } from "../../types/interp";

interface Props {
  result: LogitLensResult;
}

const CELL_W = 110;
const CELL_H = 28;
const LABEL_W = 80;
const HEADER_H = 32;

/** Clean BPE artifacts (Ġ = leading space, Ċ = newline) for display. */
function cleanToken(tok: string): string {
  return tok.replace(/Ġ/g, " ").replace(/Ċ/g, "\\n").replace(/â\u0096/g, "—");
}

export function LogitLensResults({ result }: Props) {
  const { input_tokens, layers } = result;
  const cols = input_tokens.length;
  const rows = layers.length;
  const svgW = LABEL_W + cols * CELL_W;
  const svgH = HEADER_H + rows * CELL_H;

  const allUnknown = input_tokens.every(
    (t) => t === "<unk>" || t === "[UNK]",
  );

  return (
    <div className="panel stack-sm">
      <h3>Logit Lens Heatmap</h3>
      <p className="text-secondary text-sm">
        Each cell shows the top predicted token at that layer/position. Color = probability.
      </p>
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
        <div style={{ overflowX: "auto" }}>
          <svg width={svgW} height={svgH} className="logit-lens-svg">
            {/* Column headers: input tokens */}
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

            {/* Rows: one per layer */}
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
                        width={CELL_W - 2}
                        height={CELL_H - 2}
                        rx={3}
                        fill="var(--accent)"
                        opacity={opacity}
                      />
                      <text x={CELL_W / 2} y={CELL_H / 2 + 4} textAnchor="middle" className="chart-cell-text">
                        {cleanToken(top.token)} ({(top.prob * 100).toFixed(0)}%)
                      </text>
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
        </div>
      )}
    </div>
  );
}
