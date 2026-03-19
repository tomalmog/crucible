import { useState } from "react";
import { formatSize } from "../../components/shared/RegistryRow";
import { EmptyState } from "../../components/shared/EmptyState";
import type { ClusterRemoteStorage } from "../../hooks/useResourceData";

interface RemoteStoragePanelProps {
  remoteStorage: ClusterRemoteStorage[];
}

export function RemoteStoragePanel({ remoteStorage }: RemoteStoragePanelProps) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (remoteStorage.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">
          <h3>Remote Storage</h3>
        </div>
        <EmptyState title="No clusters configured" description="Register a cluster to see remote storage usage." />
      </div>
    );
  }

  const current = remoteStorage[selectedIdx] ?? remoteStorage[0];
  const datasetsTotal = current.datasets.reduce((sum, d) => sum + d.sizeBytes, 0);
  const modelsTotal = Object.values(current.modelSizes).reduce((sum, s) => sum + s, 0);
  const total = datasetsTotal + modelsTotal;

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Remote Storage</h3>
        {remoteStorage.length > 1 && (
          <select
            value={selectedIdx}
            onChange={(e) => setSelectedIdx(Number(e.target.value))}
          >
            {remoteStorage.map((rs, i) => (
              <option key={rs.cluster.name} value={i}>
                {rs.cluster.name}
              </option>
            ))}
          </select>
        )}
        {remoteStorage.length === 1 && (
          <span className="text-xs text-tertiary">
            {current.cluster.name}
          </span>
        )}
      </div>
      <table className="overview-table">
        <tbody>
          <tr>
            <td className="overview-label">Datasets</td>
            <td className="overview-value">
              {current.datasets.length} ({formatSize(datasetsTotal)})
            </td>
          </tr>
          <tr>
            <td className="overview-label">Models</td>
            <td className="overview-value">
              {Object.keys(current.modelSizes).length} ({formatSize(modelsTotal)})
            </td>
          </tr>
          <tr>
            <td className="overview-label">Training Runs</td>
            <td className="overview-value text-tertiary">N/A</td>
          </tr>
          <tr>
            <td className="overview-label">Cache</td>
            <td className="overview-value text-tertiary">N/A</td>
          </tr>
          <tr>
            <td className="overview-label">Total Used</td>
            <td className="overview-value" style={{ fontWeight: 600 }}>
              {formatSize(total)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
