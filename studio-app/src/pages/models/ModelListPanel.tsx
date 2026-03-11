import { useState, useEffect, useMemo } from "react";
import { ChevronDown, ChevronRight, Loader2, Trash2, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
import type { ModelGroup, ModelVersion } from "../../types/models";

type DeleteScope = "local" | "remote" | "both";
type ListTab = "local" | "remote";

interface ModelListPanelProps {
  refreshKey?: number;
  onRefreshingChange?: (busy: boolean) => void;
}

export function ModelListPanel({ refreshKey, onRefreshingChange }: ModelListPanelProps) {
  const {
    dataRoot,
    modelGroups,
    selectedModelName,
    setSelectedModelName,
    modelVersions,
    selectedModel,
    setSelectedModel,
    refreshModels,
  } = useCrucible();
  const command = useCrucibleCommand();

  const [listTab, setListTab] = useState<ListTab>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [isLoadingClusters, setIsLoadingClusters] = useState(true);
  const [pendingDeleteGroup, setPendingDeleteGroup] = useState<ModelGroup | null>(null);

  // Load clusters on mount and when refresh is triggered
  useEffect(() => {
    setIsLoadingClusters(true);
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0 && !selectedCluster) setSelectedCluster(c[0].name);
    })
    .catch(console.error)
    .finally(() => {
      setIsLoadingClusters(false);
      onRefreshingChange?.(false);
    });
  }, [dataRoot, refreshKey]);

  /** Auto-determine delete scope from the model's locations. */
  function scopeForGroup(group: ModelGroup): DeleteScope {
    if (group.hasLocal && group.hasRemote) return "both";
    if (group.hasRemote) return "remote";
    return "local";
  }

  async function confirmDelete(): Promise<void> {
    if (!pendingDeleteGroup) return;
    const scope = scopeForGroup(pendingDeleteGroup);
    const args = ["model", "delete", "--name", pendingDeleteGroup.modelName, "--yes"];
    if (scope === "local" || scope === "both") args.push("--delete-local");
    if (scope === "remote" || scope === "both") args.push("--include-remote");
    await command.run(dataRoot, args);
    setPendingDeleteGroup(null);
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
          onClick={() => setListTab("local")}
        >
          Local
        </button>
        <button
          className={`tab-item ${listTab === "remote" ? "active" : ""}`}
          onClick={() => setListTab("remote")}
        >
          Remote
        </button>
      </div>

      {!isLocal && isLoadingClusters && (
        <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
          <Loader2 size={20} className="spin" />
        </div>
      )}
      {!isLocal && !isLoadingClusters && clusters.length === 0 && (
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
              onToggle={() => setSelectedModelName(
                group.modelName === selectedModelName ? null : group.modelName,
              )}
              onSelectVersion={setSelectedModel}
              onDeleteStart={() => setPendingDeleteGroup(group)}
            />
          ))}
        </div>
      )}

      {pendingDeleteGroup && (
        <DeleteModal
          modelName={pendingDeleteGroup.modelName}
          isDeleting={command.isRunning}
          onConfirm={() => confirmDelete().catch(console.error)}
          onCancel={() => setPendingDeleteGroup(null)}
        />
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
  onToggle: () => void;
  onSelectVersion: (v: ModelVersion) => void;
  onDeleteStart: () => void;
}

function ModelGroupRow({
  group, isExpanded, versions, selectedModel,
  onToggle, onSelectVersion, onDeleteStart,
}: ModelGroupRowProps) {
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
        <button
          className="btn btn-ghost btn-sm btn-icon"
          style={{ flexShrink: 0 }}
          title="Delete model"
          onClick={onDeleteStart}
        >
          <Trash2 size={12} />
        </button>
      </div>

      {isExpanded && (
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

/* ---- Delete confirmation modal ---- */

function DeleteModal(p: {
  modelName: string; isDeleting: boolean;
  onConfirm: () => void; onCancel: () => void;
}) {
  // Close on Escape
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && !p.isDeleting) p.onCancel();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [p.isDeleting, p.onCancel]);

  return (
    <div className="modal-backdrop" onClick={p.isDeleting ? undefined : p.onCancel}>
      <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">Delete Model</h3>
          {!p.isDeleting && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={p.onCancel}>
              <X size={16} />
            </button>
          )}
        </div>
        <div className="confirm-modal-body">
          <p>Are you sure you want to delete <strong>{p.modelName}</strong>? This will remove all versions and associated files.</p>
        </div>
        <div className="confirm-modal-footer">
          {!p.isDeleting && (
            <button className="btn btn-sm" onClick={p.onCancel}>Cancel</button>
          )}
          <button className="btn btn-sm btn-error" onClick={p.onConfirm} disabled={p.isDeleting}>
            {p.isDeleting ? "Deleting..." : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}
