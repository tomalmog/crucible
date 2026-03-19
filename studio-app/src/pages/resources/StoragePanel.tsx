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
  const runsTotal = isLocal ? storage.runsBytes : null;
  const cacheTotal = isLocal ? storage.cacheBytes : null;
  const totalUsed = isLocal
    ? storage.totalBytes
    : datasetsTotal + modelsTotal;
  const diskAvailable = isLocal ? storage.diskAvailableBytes : null;

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Storage</h3>
        <select value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="local">Local</option>
          {remoteStorage.map((rs) => (
            <option key={rs.cluster.name} value={rs.cluster.name}>
              {rs.cluster.name}
            </option>
          ))}
        </select>
      </div>
      <table className="overview-table">
        <tbody>
          <tr>
            <td className="overview-label">Datasets</td>
            <td className="overview-value">
              {isLocal
                ? formatSize(datasetsTotal)
                : `${cluster!.datasets.length} (${formatSize(datasetsTotal)})`}
            </td>
          </tr>
          <tr>
            <td className="overview-label">Models</td>
            <td className="overview-value">
              {isLocal
                ? formatSize(modelsTotal)
                : `${Object.keys(cluster!.modelSizes).length} (${formatSize(modelsTotal)})`}
            </td>
          </tr>
          <tr>
            <td className="overview-label">Training Runs</td>
            <td className={`overview-value${runsTotal == null ? " text-tertiary" : ""}`}>
              {runsTotal != null ? (formatSize(runsTotal) || "0 B") : "N/A"}
            </td>
          </tr>
          <tr>
            <td className="overview-label">Cache</td>
            <td className={`overview-value${cacheTotal == null ? " text-tertiary" : ""}`}>
              {cacheTotal != null ? (formatSize(cacheTotal) || "0 B") : "N/A"}
            </td>
          </tr>
          <tr>
            <td className="overview-label">Total Used</td>
            <td className="overview-value" style={{ fontWeight: 600 }}>
              {formatSize(totalUsed)}
            </td>
          </tr>
          <tr>
            <td className="overview-label">Disk Available</td>
            <td className={`overview-value${diskAvailable == null ? " text-tertiary" : ""}`}>
              {diskAvailable != null ? formatSize(diskAvailable) : "N/A"}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
