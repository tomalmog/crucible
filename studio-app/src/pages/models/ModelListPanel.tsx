import { useForge } from "../../context/ForgeContext";

export function ModelListPanel() {
  const { modelVersions, selectedModel, setSelectedModel } = useForge();

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Model Versions</h3>
      {modelVersions.length === 0 ? (
        <p className="text-tertiary">No models registered yet.</p>
      ) : (
        <div>
          {modelVersions.map((v) => (
            <button
              key={v.versionId}
              className={`nav-item ${v.versionId === selectedModel?.versionId ? "active" : ""}`}
              onClick={() => setSelectedModel(v)}
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
