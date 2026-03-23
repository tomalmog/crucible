import { useState } from "react";
import { formatSize } from "../../components/shared/RegistryRow";
import type { StorageBreakdown } from "../../types/resources";
import type { ClusterRemoteStorage } from "../../hooks/useResourceData";

interface StoragePanelProps {
  storage: StorageBreakdown | null;
  remoteStorage: ClusterRemoteStorage[];
}

export function StoragePanel({ storage, remoteStorage }: StoragePanelProps) {
  const [selected, setSelected] = useState("local");

  if (!storage) return null;

  const isLocal = selected === "local";
  const cluster = !isLocal
    ? remoteStorage.find((rs) => rs.cluster.name === selected)
    : null;

  const datasetsTotal = cluster
    ? cluster.datasets.reduce((sum, d) => sum + d.sizeBytes, 0)
    : storage.datasetsBytes;
  const modelsTotal = cluster
    ? Object.values(cluster.modelSizes).reduce((sum, s) => sum + s, 0)
    : storage.modelsBytes;
  const runsTotal = isLocal ? storage.runsBytes : 0;
  const cacheTotal = isLocal ? storage.cacheBytes : 0;
  const totalUsed = isLocal
    ? storage.totalBytes
    : datasetsTotal + modelsTotal;
  const diskAvailable = isLocal ? storage.diskAvailableBytes : null;

  // Build bar rows
  const bars: { label: string; bytes: number; color?: string }[] = [
    { label: "Datasets", bytes: datasetsTotal },
    { label: "Models", bytes: modelsTotal },
  ];
  if (isLocal) {
    bars.push({ label: "Training Runs", bytes: runsTotal });
    bars.push({ label: "Cache", bytes: cacheTotal });
  }

  const maxBytes = Math.max(...bars.map((b) => b.bytes), 1);

  return (
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Storage</h3>
        {remoteStorage.length > 0 && (
          <select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ width: "auto", minWidth: 100, padding: "4px 8px", fontSize: "0.75rem" }}>
            <option value="local">Local</option>
            {remoteStorage.map((rs) => (
              <option key={rs.cluster.name} value={rs.cluster.name}>
                {rs.cluster.name}
              </option>
            ))}
          </select>
        )}
      </div>

      <div style={{ display: "grid", gap: 8 }}>
        {bars.map((b) => (
          <div className="storage-bar-row" key={b.label}>
            <div className="storage-bar-label">
              <span className="storage-bar-label-name">{b.label}</span>
              <span className="storage-bar-label-value">{formatSize(b.bytes) || "0 B"}</span>
            </div>
            <div className="storage-bar-track">
              <div
                className="storage-bar-fill"
                style={{ width: `${Math.max((b.bytes / maxBytes) * 100, 0.5)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="storage-total">
        <span className="storage-total-label">Total Used</span>
        <span className="storage-total-value">{formatSize(totalUsed)}</span>
      </div>

      {diskAvailable != null && (
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "var(--text-tertiary)" }}>
          <span>Disk Available</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{formatSize(diskAvailable)}</span>
        </div>
      )}
    </div>
  );
}
