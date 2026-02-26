import type { ModelVersion } from "../../types/models";

interface ModelListPanelProps {
  versions: ModelVersion[];
  selected: ModelVersion | null;
  onSelect: (version: ModelVersion) => void;
}

export function ModelListPanel({ versions, selected, onSelect }: ModelListPanelProps) {
  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Model Versions</h3>
      {versions.length === 0 ? (
        <p className="text-tertiary">No models registered yet.</p>
      ) : (
        <div>
          {versions.map((v) => (
            <button
              key={v.versionId}
              className={`nav-item ${v.versionId === selected?.versionId ? "active" : ""}`}
              onClick={() => onSelect(v)}
            >
              <div className="model-version-row">
                <span className="text-mono text-sm">{v.versionId.slice(0, 16)}...</span>
                <span className="model-version-meta">
                  <span className="text-xs text-tertiary">{v.createdAt}</span>
                  {v.isActive && <span className="badge badge-accent">active</span>}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
