import type { ReactNode } from "react";
import type { LinearProbeResult, ProbeLayerResult } from "../../types/interp";
import { EvidenceSummary } from "./EvidenceSummary";
import { formatPercent } from "./interpDisplay";

interface Props {
  result: LinearProbeResult;
}

export function LinearProbeResults({ result }: Props): ReactNode {
  const { layers } = result;
  const bestLayer = findBestLayer(layers);

  return (
    <div className="interp-evidence-card stack-md">
      <div className="interp-evidence-header">
        <div>
          <span className="interp-kicker">Linear probe</span>
          <h3>Classification accuracy by layer</h3>
          <p>
            A probe scans layers for representations that separate target labels.
            Peaks mark layers where a simple classifier recovers the labels most
            reliably.
          </p>
        </div>
        <span className="interp-evidence-badge">{layers.length} layers</span>
      </div>
      <EvidenceSummary
        items={[
          { label: "Best layer", value: bestLayer ? `L${bestLayer.layer_index}` : "n/a", tone: "positive" },
          { label: "Best accuracy", value: bestLayer ? formatPercent(bestLayer.accuracy) : "n/a" },
          { label: "Classes", value: bestLayer ? String(bestLayer.num_classes) : "n/a" },
          { label: "Probe sweep", value: `${layers.length} layer${layers.length === 1 ? "" : "s"}` },
        ]}
      />
      <ProbeHeatmap bestLayer={bestLayer} layers={layers} />
      {bestLayer && bestLayer.confusion_matrix.length > 0 && (
        <ConfusionMatrix layer={bestLayer} />
      )}
    </div>
  );
}

function findBestLayer(layers: ProbeLayerResult[]): ProbeLayerResult | null {
  let bestLayer: ProbeLayerResult | null = null;
  for (const layer of layers) {
    if (!bestLayer || layer.accuracy > bestLayer.accuracy) bestLayer = layer;
  }
  return bestLayer;
}

function ProbeHeatmap({
  bestLayer,
  layers,
}: {
  bestLayer: ProbeLayerResult | null;
  layers: ProbeLayerResult[];
}): ReactNode {
  return (
    <div className="probe-heatmap">
      <span className="metric-label">Layer separability scan</span>
      <div className="probe-heatmap-grid">
        {layers.map((layer) => {
          const strength = Math.min(1, Math.max(0, layer.accuracy));
          const isBest = bestLayer?.layer_name === layer.layer_name;
          return (
            <article
              className={`probe-heatmap-cell${isBest ? " best" : ""}`}
              key={layer.layer_name}
              style={{ opacity: 0.28 + strength * 0.72 }}
              title={`${layer.layer_name}: ${formatPercent(layer.accuracy)}`}
            >
              <span>L{layer.layer_index}</span>
              <strong>{formatPercent(layer.accuracy, 0)}</strong>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function ConfusionMatrix({ layer }: { layer: ProbeLayerResult }): ReactNode {
  const { class_names, confusion_matrix } = layer;
  return (
    <div className="confusion-matrix-card">
      <div className="panel-header">
        <div>
          <h4>Best-layer confusion matrix</h4>
          <p className="text-secondary text-sm">Layer {layer.layer_index} · {layer.layer_name}</p>
        </div>
      </div>
      <div className="confusion-table-scroll">
        <table className="confusion-table">
          <thead>
            <tr>
              <th>true \ pred</th>
              {class_names.map((name) => <th key={`head-${name}`}>{name}</th>)}
            </tr>
          </thead>
          <tbody>
            {confusion_matrix.map((row, rowIndex) => (
              <tr key={class_names[rowIndex]}>
                <th>{class_names[rowIndex]}</th>
                {row.map((value, colIndex) => (
                  <td className={colIndex === rowIndex ? "diagonal" : ""} key={`${rowIndex}-${colIndex}`}>
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
