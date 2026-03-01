import { ChevronDown, ChevronRight } from "lucide-react";
import { useForge } from "../../context/ForgeContext";

export function ModelListPanel() {
  const {
    modelGroups,
    selectedModelName,
    setSelectedModelName,
    modelVersions,
    selectedModel,
    setSelectedModel,
  } = useForge();

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Models</h3>
      {modelGroups.length === 0 ? (
        <p className="text-tertiary">No models registered yet.</p>
      ) : (
        <div>
          {modelGroups.map((group) => {
            const isExpanded = group.modelName === selectedModelName;
            return (
              <div key={group.modelName}>
                <button
                  className={`nav-item ${isExpanded ? "active" : ""}`}
                  onClick={() =>
                    setSelectedModelName(isExpanded ? null : group.modelName)
                  }
                >
                  <span style={{ display: "flex", alignItems: "center", gap: 6, flex: 1 }}>
                    {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                    <span>{group.modelName}</span>
                  </span>
                  <span className="badge">{group.versionCount}</span>
                </button>
                {isExpanded && (
                  <div>
                    {modelVersions.map((v) => (
                      <button
                        key={v.versionId}
                        className={`nav-item nav-item-nested ${v.versionId === selectedModel?.versionId ? "active" : ""}`}
                        onClick={() => setSelectedModel(v)}
                      >
                        <div className="model-version-row">
                          <span className="text-mono text-sm">
                            {v.versionId.slice(0, 16)}...
                          </span>
                          <span className="model-version-meta">
                            <span className="text-xs text-tertiary">
                              {v.createdAt}
                            </span>
                            {v.isActive && (
                              <span className="badge badge-accent">active</span>
                            )}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
