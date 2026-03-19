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
    <div className="panel">
      <div className="panel-header">
        <h3>Cleanup</h3>
      </div>

      {/* Orphaned runs */}
      <div style={{ padding: "0 16px 16px" }}>
        <div className="flex-row" style={{ marginBottom: 8 }}>
          <span className="text-sm" style={{ fontWeight: 600 }}>
            Orphaned Runs ({orphans.length})
          </span>
          {orphanTotal > 0 && (
            <span className="text-xs text-tertiary">{formatSize(orphanTotal)} reclaimable</span>
          )}
        </div>

        {orphans.length === 0 ? (
          <p className="text-sm text-tertiary">No orphaned runs found.</p>
        ) : (
          <>
            <div style={{ maxHeight: 200, overflow: "auto", marginBottom: 8 }}>
              {orphans.map((o) => (
                <label
                  key={o.runId}
                  className="flex-row"
                  style={{ marginBottom: 4, cursor: "pointer", gap: 8 }}
                >
                  <input
                    type="checkbox"
                    checked={selected.has(o.runId)}
                    onChange={() => toggleSelect(o.runId)}
                  />
                  <span className="text-sm" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {o.runId}
                  </span>
                  <span className="text-xs text-tertiary">{o.datasetName}</span>
                  <span className="text-xs text-tertiary">{formatSize(o.sizeBytes)}</span>
                </label>
              ))}
            </div>
            <div className="flex-row" style={{ gap: 8 }}>
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
      <div style={{ padding: "0 16px 16px", borderTop: "1px solid var(--border)" }}>
        <div className="flex-row" style={{ marginTop: 16, marginBottom: 8 }}>
          <span className="text-sm" style={{ fontWeight: 600 }}>Cache</span>
          <span className="text-xs text-tertiary">
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
