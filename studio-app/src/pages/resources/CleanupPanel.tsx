import { useState } from "react";
import { Trash2, Loader2 } from "lucide-react";
import { formatSize } from "../../components/shared/RegistryRow";
import { useCrucible } from "../../context/CrucibleContext";
import { deleteOrphanedRuns, clearCache } from "../../api/resourcesApi";
import type { StorageBreakdown, OrphanedRun } from "../../types/resources";

interface CleanupPanelProps {
  orphans: OrphanedRun[];
  storage: StorageBreakdown | null;
  onRefresh: () => void;
}

export function CleanupPanel({ orphans, storage, onRefresh }: CleanupPanelProps) {
  const { dataRoot } = useCrucible();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);

  function toggleSelect(runId: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(runId)) next.delete(runId);
      else next.add(runId);
      return next;
    });
  }

  async function handleDeleteSelected() {
    if (selected.size === 0) return;
    setDeleting(true);
    try {
      await deleteOrphanedRuns(dataRoot, [...selected]);
      setSelected(new Set());
      onRefresh();
    } catch (err) {
      console.error("Failed to delete orphaned runs:", err);
    } finally {
      setDeleting(false);
    }
  }

  async function handleDeleteAll() {
    if (!confirmDeleteAll) {
      setConfirmDeleteAll(true);
      return;
    }
    setDeleting(true);
    try {
      await deleteOrphanedRuns(dataRoot, orphans.map((o) => o.runId));
      setSelected(new Set());
      setConfirmDeleteAll(false);
      onRefresh();
    } catch (err) {
      console.error("Failed to delete orphaned runs:", err);
    } finally {
      setDeleting(false);
    }
  }

  async function handleClearCache() {
    setClearing(true);
    try {
      await clearCache(dataRoot);
      onRefresh();
    } catch (err) {
      console.error("Failed to clear cache:", err);
    } finally {
      setClearing(false);
    }
  }

  const orphanTotal = orphans.reduce((sum, o) => sum + o.sizeBytes, 0);

  return (
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Cleanup</h3>
        {orphanTotal > 0 && (
          <span className="badge">{formatSize(orphanTotal)} reclaimable</span>
        )}
      </div>

      {/* Orphaned runs */}
      <div>
        <div className="resource-section-header" style={{ marginBottom: 4 }}>
          <span className="resource-section-title">Orphaned Runs ({orphans.length})</span>
        </div>

        {orphans.length === 0 ? (
          <p className="text-sm text-tertiary" style={{ margin: 0 }}>No orphaned runs found.</p>
        ) : (
          <>
            <div style={{ maxHeight: 200, overflow: "auto" }}>
              {orphans.map((o) => (
                <label key={o.runId} className="orphan-row" style={{ cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={selected.has(o.runId)}
                    onChange={() => toggleSelect(o.runId)}
                    style={{ width: 16, height: 16, flexShrink: 0 }}
                  />
                  <span className="orphan-row-name">{o.runId}</span>
                  <span className="orphan-row-meta">{o.datasetName}</span>
                  <span className="orphan-row-meta">{formatSize(o.sizeBytes)}</span>
                </label>
              ))}
            </div>
            <div className="flex-row" style={{ gap: 8, marginTop: 8 }}>
              <button
                className="btn btn-sm"
                onClick={handleDeleteSelected}
                disabled={selected.size === 0 || deleting}
              >
                {deleting && selected.size > 0 ? (
                  <Loader2 size={12} className="spin" />
                ) : (
                  <Trash2 size={12} />
                )}
                Delete Selected ({selected.size})
              </button>
              <button
                className="btn btn-sm btn-error"
                onClick={handleDeleteAll}
                disabled={deleting}
                onBlur={() => setConfirmDeleteAll(false)}
              >
                {deleting && confirmDeleteAll ? (
                  <Loader2 size={12} className="spin" />
                ) : (
                  <Trash2 size={12} />
                )}
                {confirmDeleteAll ? "Confirm Delete All?" : `Delete All (${orphans.length})`}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Cache */}
      <div className="resource-section">
        <div className="resource-section-header">
          <span className="resource-section-title">Cache</span>
          <span className="text-xs text-tertiary" style={{ fontFamily: "var(--font-mono)" }}>
            {storage ? formatSize(storage.cacheBytes) : "..."}
          </span>
        </div>
        <button className="btn btn-sm" onClick={handleClearCache} disabled={clearing}>
          {clearing ? <Loader2 size={12} className="spin" /> : <Trash2 size={12} />}
          Clear Cache
        </button>
      </div>
    </div>
  );
}
