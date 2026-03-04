import { CheckCircle, XCircle, Trash2, RefreshCw } from "lucide-react";
import type { ClusterConfig } from "../../types/remote";

interface ClusterCardProps {
  cluster: ClusterConfig;
  onRemove: () => void;
  onValidate: () => void;
}

export function ClusterCard({ cluster, onRemove, onValidate }: ClusterCardProps) {
  const isValidated = !!cluster.validatedAt;

  return (
    <div className="panel">
      <div className="run-row-header">
        <div className="flex-row">
          <span className="run-row-id">{cluster.name}</span>
          {isValidated ? (
            <span className="badge badge-success">
              <CheckCircle size={12} /> Validated
            </span>
          ) : (
            <span className="badge badge-error">
              <XCircle size={12} /> Not validated
            </span>
          )}
        </div>
        <div className="flex-row">
          <button className="btn btn-sm" onClick={onValidate} title="Validate cluster">
            <RefreshCw size={12} /> Validate
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onRemove} title="Remove cluster">
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      <table className="overview-table">
        <tbody>
          <tr>
            <td className="overview-label">Host</td>
            <td className="overview-value">{cluster.user}@{cluster.host}</td>
          </tr>
          {cluster.defaultPartition && (
            <tr>
              <td className="overview-label">Default Partition</td>
              <td className="overview-value">{cluster.defaultPartition}</td>
            </tr>
          )}
          {cluster.partitions.length > 0 && (
            <tr>
              <td className="overview-label">Partitions</td>
              <td className="overview-value">{cluster.partitions.join(", ")}</td>
            </tr>
          )}
          {cluster.gpuTypes.length > 0 && (
            <tr>
              <td className="overview-label">GPU Types</td>
              <td className="overview-value">{cluster.gpuTypes.join(", ")}</td>
            </tr>
          )}
          <tr>
            <td className="overview-label">Python</td>
            <td className="overview-value">{cluster.pythonPath}</td>
          </tr>
          <tr>
            <td className="overview-label">Workspace</td>
            <td className="overview-value">{cluster.remoteWorkspace}</td>
          </tr>
          {isValidated && (
            <tr>
              <td className="overview-label">Last Validated</td>
              <td className="overview-value">{cluster.validatedAt}</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
