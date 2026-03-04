import { useState } from "react";
import { ChevronDown, ChevronRight, Trash2 } from "lucide-react";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";

type DeleteScope = "registry" | "local" | "remote" | "both";

export function ModelListPanel() {
  const {
    dataRoot,
    modelGroups,
    selectedModelName,
    setSelectedModelName,
    modelVersions,
    selectedModel,
    setSelectedModel,
    refreshModels,
  } = useForge();
  const command = useForgeCommand();
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleteScope, setDeleteScope] = useState<DeleteScope>("local");

  function cancelDelete() {
    setPendingDelete(null);
    setDeleteScope("local");
  }

  async function confirmDelete(modelName: string) {
    const args = ["model", "delete", "--name", modelName, "--yes"];
    if (deleteScope === "local" || deleteScope === "both") args.push("--delete-local");
    if (deleteScope === "remote" || deleteScope === "both") args.push("--include-remote");
    if (deleteScope === "remote") args.push("--keep-registry");
    await command.run(dataRoot, args);
    cancelDelete();
    await refreshModels();
  }

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Models</h3>
      {modelGroups.length === 0 ? (
        <p className="text-tertiary">No models registered yet.</p>
      ) : (
        <div>
          {modelGroups.map((group) => {
            const isExpanded = group.modelName === selectedModelName;
            const isDeleting = pendingDelete === group.modelName;
            return (
              <div key={group.modelName}>
                <div className="flex-row" style={{ alignItems: "center" }}>
                  <button
                    className={`nav-item ${isExpanded ? "active" : ""}`}
                    style={{ flex: 1 }}
                    onClick={() => {
                      setSelectedModelName(isExpanded ? null : group.modelName);
                      cancelDelete();
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: 6, flex: 1 }}>
                      {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                      <span>{group.modelName}</span>
                    </span>
                    <span className="badge">{group.versionCount}</span>
                  </button>
                  {!isDeleting && (
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      style={{ flexShrink: 0 }}
                      title="Delete model"
                      onClick={() => setPendingDelete(group.modelName)}
                    >
                      <Trash2 size={12} />
                    </button>
                  )}
                </div>

                {isDeleting && (
                  <div style={{ padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
                    <span className="text-sm" style={{ color: "var(--error)" }}>
                      Delete "{group.modelName}"?
                    </span>
                    <select
                      className="text-sm"
                      value={deleteScope}
                      onChange={(e) => setDeleteScope(e.currentTarget.value as DeleteScope)}
                    >
                      <option value="registry">Registry only (keep files)</option>
                      <option value="local">Local files</option>
                      <option value="remote">Remote files</option>
                      <option value="both">Local + Remote</option>
                    </select>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button
                        className="btn btn-sm btn-error"
                        onClick={() => confirmDelete(group.modelName).catch(console.error)}
                        disabled={command.isRunning}
                      >
                        {command.isRunning ? "Deleting..." : "Confirm"}
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={cancelDelete}
                        disabled={command.isRunning}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {isExpanded && !isDeleting && (
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
