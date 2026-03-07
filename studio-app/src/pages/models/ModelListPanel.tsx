import { useState, useEffect, useMemo } from "react";
import { ChevronDown, ChevronRight, Trash2 } from "lucide-react";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
import type { ModelGroup, ModelVersion } from "../../types/models";

type DeleteScope = "registry" | "local" | "remote" | "both";
type ListTab = "local" | "remote";

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

  const [listTab, setListTab] = useState<ListTab>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleteScope, setDeleteScope] = useState<DeleteScope>("local");

  // Load clusters on mount
  useEffect(() => {
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0) setSelectedCluster(c[0].name);
    }).catch(console.error);
  }, [dataRoot]);

  function cancelDelete(): void {
    setPendingDelete(null);
    setDeleteScope("local");
  }

  function switchTab(tab: ListTab): void {
    setListTab(tab);
    cancelDelete();
  }

  async function confirmDelete(modelName: string): Promise<void> {
    const args = ["model", "delete", "--name", modelName, "--yes"];
    if (deleteScope === "local" || deleteScope === "both") args.push("--delete-local");
    if (deleteScope === "remote" || deleteScope === "both") args.push("--include-remote");
    if (deleteScope === "remote") args.push("--keep-registry");
    await command.run(dataRoot, args);
    cancelDelete();
    await refreshModels();
  }

  const isLocal = listTab === "local";

  // Filter groups by tab: local shows groups with local versions, remote shows groups with remote
  const filteredGroups = useMemo(() =>
    modelGroups.filter((g) => isLocal ? g.hasLocal : g.hasRemote),
    [modelGroups, isLocal],
  );

  // For remote tab, filter versions to only remote/both for the selected cluster
  const filteredVersions = useMemo(() => {
    if (isLocal) return modelVersions;
    return modelVersions.filter((v) => {
      const isRemoteVersion = v.locationType === "remote" || v.locationType === "both";
      if (!selectedCluster) return isRemoteVersion;
      return isRemoteVersion && v.remoteHost === selectedCluster;
    });
  }, [modelVersions, isLocal, selectedCluster]);

  const emptyMsg = isLocal
    ? "No local models registered yet."
    : "No remote models found.";

  return (
    <div className="panel overflow-auto">
      <h3 className="panel-title">Models</h3>

      <div className="tab-list">
        <button
          className={`tab-item ${listTab === "local" ? "active" : ""}`}
          onClick={() => switchTab("local")}
        >
          Local
        </button>
        <button
          className={`tab-item ${listTab === "remote" ? "active" : ""}`}
          onClick={() => switchTab("remote")}
        >
          Remote
        </button>
      </div>

      {!isLocal && clusters.length === 0 && (
        <p className="text-tertiary">No clusters registered.</p>
      )}

      {!isLocal && clusters.length > 0 && (
        <select
          className="input"
          value={selectedCluster}
          onChange={(e) => setSelectedCluster(e.target.value)}
          style={{ marginBottom: 8, width: "100%" }}
        >
          <option value="">All clusters</option>
          {clusters.map((c) => (
            <option key={c.name} value={c.name}>{c.name}</option>
          ))}
        </select>
      )}

      {filteredGroups.length === 0 ? (
        <p className="text-tertiary">{emptyMsg}</p>
      ) : (
        <div>
          {filteredGroups.map((group) => (
            <ModelGroupRow
              key={group.modelName}
              group={group}
              isExpanded={group.modelName === selectedModelName}
              versions={filteredVersions}
              selectedModel={selectedModel}
              pendingDelete={pendingDelete}
              deleteScope={deleteScope}
              isDeleting={command.isRunning}
              onToggle={() => {
                setSelectedModelName(
                  group.modelName === selectedModelName ? null : group.modelName,
                );
                cancelDelete();
              }}
              onSelectVersion={setSelectedModel}
              onDeleteStart={() => setPendingDelete(group.modelName)}
              onDeleteConfirm={() => confirmDelete(group.modelName).catch(console.error)}
              onDeleteCancel={cancelDelete}
              onDeleteScopeChange={setDeleteScope}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- Model group with expandable versions ---- */

interface ModelGroupRowProps {
  group: ModelGroup;
  isExpanded: boolean;
  versions: ModelVersion[];
  selectedModel: ModelVersion | null;
  pendingDelete: string | null;
  deleteScope: DeleteScope;
  isDeleting: boolean;
  onToggle: () => void;
  onSelectVersion: (v: ModelVersion) => void;
  onDeleteStart: () => void;
  onDeleteConfirm: () => void;
  onDeleteCancel: () => void;
  onDeleteScopeChange: (scope: DeleteScope) => void;
}

function ModelGroupRow({
  group, isExpanded, versions, selectedModel, pendingDelete,
  deleteScope, isDeleting, onToggle, onSelectVersion,
  onDeleteStart, onDeleteConfirm, onDeleteCancel, onDeleteScopeChange,
}: ModelGroupRowProps) {
  const isDeleteTarget = pendingDelete === group.modelName;

  return (
    <div>
      <div className="flex-row" style={{ alignItems: "center" }}>
        <button
          className={`nav-item ${isExpanded ? "active" : ""}`}
          style={{ flex: 1 }}
          onClick={onToggle}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 6, flex: 1 }}>
            {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span>{group.modelName}</span>
          </span>
          <span className="badge">{group.versionCount}</span>
        </button>
        {!isDeleteTarget && (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            style={{ flexShrink: 0 }}
            title="Delete model"
            onClick={onDeleteStart}
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>

      {isDeleteTarget && (
        <DeleteConfirmation
          modelName={group.modelName}
          deleteScope={deleteScope}
          isDeleting={isDeleting}
          onScopeChange={onDeleteScopeChange}
          onConfirm={onDeleteConfirm}
          onCancel={onDeleteCancel}
        />
      )}

      {isExpanded && !isDeleteTarget && (
        <div>
          {versions.map((v) => (
            <VersionRow
              key={v.versionId}
              version={v}
              isSelected={v.versionId === selectedModel?.versionId}
              onSelect={() => onSelectVersion(v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- Version row ---- */

function VersionRow({ version, isSelected, onSelect }: {
  version: ModelVersion;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const locationBadge = version.locationType === "remote" ? "remote"
    : version.locationType === "both" ? "local + remote" : null;

  return (
    <button
      className={`nav-item nav-item-nested ${isSelected ? "active" : ""}`}
      onClick={onSelect}
    >
      <div className="model-version-row">
        <span className="text-mono text-sm">
          {version.versionId.slice(0, 16)}...
        </span>
        <span className="model-version-meta">
          <span className="text-xs text-tertiary">
            {version.createdAt}
          </span>
          {version.isActive && (
            <span className="badge badge-accent">active</span>
          )}
          {locationBadge && (
            <span className="badge">{locationBadge}</span>
          )}
        </span>
      </div>
    </button>
  );
}

/* ---- Delete confirmation ---- */

function DeleteConfirmation(p: {
  modelName: string; deleteScope: DeleteScope; isDeleting: boolean;
  onScopeChange: (s: DeleteScope) => void; onConfirm: () => void; onCancel: () => void;
}) {
  return (
    <div style={{ padding: "8px 12px", display: "flex", flexDirection: "column", gap: 6 }}>
      <span className="text-sm" style={{ color: "var(--error)" }}>
        Delete &quot;{p.modelName}&quot;?
      </span>
      <select className="text-sm" value={p.deleteScope}
        onChange={(e) => p.onScopeChange(e.currentTarget.value as DeleteScope)}>
        <option value="registry">Registry only (keep files)</option>
        <option value="local">Local files</option>
        <option value="remote">Remote files</option>
        <option value="both">Local + Remote</option>
      </select>
      <div style={{ display: "flex", gap: 6 }}>
        <button className="btn btn-sm btn-error" onClick={p.onConfirm} disabled={p.isDeleting}>
          {p.isDeleting ? "Deleting..." : "Confirm"}
        </button>
        <button className="btn btn-sm" onClick={p.onCancel} disabled={p.isDeleting}>Cancel</button>
      </div>
    </div>
  );
}
