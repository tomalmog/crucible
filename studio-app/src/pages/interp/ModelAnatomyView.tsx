import { type ReactNode, useState } from "react";
import { formatNumber } from "./interpDisplay";
import type { InterpTab } from "./interpTabs";
import type { ModelAnatomyData, ModelAnatomyLayer, ModelLayerEvidence } from "./modelAnatomyTypes";

interface ModelAnatomyViewProps {
  data: ModelAnatomyData | null;
  onSelect: (tab: InterpTab) => void;
}

const LANDMARK_COUNT = 5;

export function ModelAnatomyView({ data, onSelect }: ModelAnatomyViewProps): ReactNode {
  const [activeLayer, setActiveLayer] = useState(0);
  if (!data) return <EmptyModelAnatomy />;

  const selectedLayer = data.layers[Math.min(activeLayer, data.layers.length - 1)];
  const evidenceCount = countEvidence(data.layers);
  const coveredLayerCount = data.layers.filter((layer) => layer.evidence.length > 0).length;
  const coverageLabel = `${coveredLayerCount}/${data.layerCount}`;
  const hasMoeMetadata = data.layers.some((layer) => layer.kind === "moe");
  const landmarks = makeLandmarks(data.layerCount);

  return (
    <section className="model-anatomy-card" aria-label="Model anatomy">
      <div className="model-anatomy-topline">
        <div>
          <span className="interp-kicker">Selected model</span>
          <h2>{data.modelName}</h2>
        </div>
        <div className="model-anatomy-status">{data.statusLabel}</div>
      </div>

      <div className="model-anatomy-board">
        <div className="model-stream-header">
          <div>
            <span className="metric-label">Residual stream</span>
            <strong>{data.hiddenSize ? `${data.hiddenSize.toLocaleString()} dim` : "unknown dim"}</strong>
          </div>
          <div>
            <span className="metric-label">Evidence coverage</span>
            <strong>{coverageLabel} layers</strong>
          </div>
          <div>
            <span className="metric-label">Completed artifacts</span>
            <strong>{evidenceCount}</strong>
          </div>
        </div>
        <div className="model-stream-rail" aria-hidden="true">
          <span>tokens</span>
          <i />
          <span>blocks</span>
          <i />
          <span>logits</span>
        </div>
        <div className="model-layer-ribbon" aria-label="Transformer layer ribbon">
          {data.layers.map((layer) => (
            <LayerButton
              isLandmark={landmarks.has(layer.index)}
              isSelected={layer.index === selectedLayer.index}
              key={layer.index}
              layer={layer}
              onSelect={() => setActiveLayer(layer.index)}
            />
          ))}
        </div>
        <div className="model-layer-axis" aria-hidden="true">
          {Array.from(landmarks).map((index) => (
            <span key={index}>L{index}</span>
          ))}
        </div>
      </div>

      <div className="model-anatomy-inspector">
        <div className="selected-layer-panel">
          <span className="metric-label">Selected layer</span>
          <strong>{selectedLayer.label}</strong>
          <p>{selectedLayer.detail}</p>
        </div>
        <LayerEvidence layer={selectedLayer} />
        <div className="model-routing-actions">
          <span className="metric-label">Route next analysis</span>
          <button type="button" className="btn" onClick={() => onSelect("logit-lens")}>logit lens</button>
          <button type="button" className="btn" onClick={() => onSelect("activation-patching")}>patching</button>
          <button type="button" className="btn" onClick={() => onSelect("sae")}>SAE</button>
        </div>
      </div>

      <div className="model-anatomy-footer">
        <div className="model-anatomy-legend">
          <span><i className="legend-swatch block" />Transformer block</span>
          {hasMoeMetadata && <span><i className="legend-swatch moe" />MoE metadata</span>}
          <span><i className="legend-swatch evidence" />Completed interp job</span>
        </div>
        <div className="model-anatomy-metrics">
          <span>{data.layerCount} layers</span>
          <span>{data.parameterLabel}</span>
          <span>{data.attentionHeads ? `${data.attentionHeads} heads` : "heads unknown"}</span>
        </div>
      </div>
    </section>
  );
}

function LayerButton({
  isLandmark,
  isSelected,
  layer,
  onSelect,
}: {
  isLandmark: boolean;
  isSelected: boolean;
  layer: ModelAnatomyLayer;
  onSelect: () => void;
}): ReactNode {
  return (
    <button
      type="button"
      aria-pressed={isSelected}
      className={`model-layer-button ${layer.kind}${layer.evidence.length > 0 ? " has-evidence" : ""}${isSelected ? " selected" : ""}`}
      title={`${layer.label} · ${layer.detail}`}
      onClick={onSelect}
    >
      {isLandmark && <span className="model-layer-index">{layer.index}</span>}
      {layer.evidence.length > 0 && (
        <span className="model-layer-evidence-mark" aria-hidden="true">
          {layer.evidence.slice(0, 3).map((item) => (
            <i key={`${item.jobId}-${item.jobType}`} />
          ))}
        </span>
      )}
    </button>
  );
}

function LayerEvidence({ layer }: { layer: ModelAnatomyLayer }): ReactNode {
  if (layer.evidence.length === 0) {
    return (
      <div className="model-evidence-empty">
        <span className="metric-label">Layer evidence</span>
        <p>No completed interp artifacts mapped to this layer yet.</p>
      </div>
    );
  }
  return (
    <div className="model-evidence-ledger">
      <span className="metric-label">Layer evidence</span>
      <table>
        <tbody>
          {layer.evidence.slice(0, 4).map((item) => (
            <EvidenceRow item={item} key={`${item.jobId}-${item.jobType}`} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EvidenceRow({ item }: { item: ModelLayerEvidence }): ReactNode {
  return (
    <tr>
      <th>{item.jobType}</th>
      <td>{item.metric === undefined ? "artifact" : formatNumber(item.metric, 3)}</td>
    </tr>
  );
}

function EmptyModelAnatomy(): ReactNode {
  return (
    <section className="model-anatomy-card model-anatomy-empty" aria-label="Model anatomy">
      <span className="interp-kicker">Selected model</span>
      <h2>No model selected</h2>
      <p className="text-secondary text-sm">
        Select a registered model to load architecture metadata and completed
        interpretability artifacts.
      </p>
    </section>
  );
}

function makeLandmarks(layerCount: number): Set<number> {
  if (layerCount <= LANDMARK_COUNT) {
    return new Set(Array.from({ length: layerCount }, (_, index) => index));
  }
  return new Set(Array.from({ length: LANDMARK_COUNT }, (_, index) => (
    Math.round((index / (LANDMARK_COUNT - 1)) * (layerCount - 1))
  )));
}

function countEvidence(layers: ModelAnatomyLayer[]): number {
  return layers.reduce((total, layer) => total + layer.evidence.length, 0);
}
