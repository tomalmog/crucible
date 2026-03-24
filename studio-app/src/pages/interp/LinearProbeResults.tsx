import type { LinearProbeResult, ProbeLayerResult } from "../../types/interp";

interface Props {
  result: LinearProbeResult;
}

export function LinearProbeResults({ result }: Props) {
  const { layers } = result;
  const maxAccuracy = Math.max(...layers.map((l) => l.accuracy), 0.01);

  return (
    <div className="panel stack-sm">
      <h3>Linear Probe Results</h3>
      <div>
        {layers.map((layer) => {
          const pct = Math.round((layer.accuracy / maxAccuracy) * 100);
          return (
            <div className="bar-row" key={layer.layer_name}>
              <div className="bar-label">
                <span>L{layer.layer_index}</span>
                <strong>{(layer.accuracy * 100).toFixed(1)}%</strong>
              </div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${pct}%` }} />
              </div>
            </div>
          );
        })}
      </div>
      {layers.length === 1 && layers[0].confusion_matrix.length > 0 && (
        <ConfusionMatrix layer={layers[0]} />
      )}
    </div>
  );
}

function ConfusionMatrix({ layer }: { layer: ProbeLayerResult }) {
  const { class_names, confusion_matrix } = layer;
  return (
    <div className="docs-table-wrap" style={{ marginTop: 12 }}>
      <table className="docs-table">
        <thead>
          <tr>
            <th>True \ Pred</th>
            {class_names.map((n) => <th key={n}>{n}</th>)}
          </tr>
        </thead>
        <tbody>
          {confusion_matrix.map((row, i) => (
            <tr key={class_names[i]}>
              <td style={{ fontWeight: 500 }}>{class_names[i]}</td>
              {row.map((val, j) => (
                <td key={j} style={i === j ? { fontWeight: 600, color: "var(--accent)" } : undefined}>
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
