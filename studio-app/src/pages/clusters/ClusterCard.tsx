import { useState } from "react";
import { CheckCircle, XCircle, Trash2, RefreshCw, RotateCcw, Loader, Pencil } from "lucide-react";
import type { ClusterConfig, ClusterBackend } from "../../types/remote";

interface ClusterCardProps {
  cluster: ClusterConfig;
  onRemove: () => void;
  onValidate: (onProgress: (output: string) => void) => Promise<void>;
  onResetEnv: () => Promise<void>;
  onEdit: () => void;
}

function truncateGpuName(name: string): string {
  if (name.length <= 18) return name;
  return name.slice(0, 16) + "…";
}

function formatValidatedDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  } catch {
    return iso;
  }
}

function getBackendLabel(backend: ClusterBackend, dockerImage: string): string {
  if (backend === "ssh") return dockerImage ? "SSH (Docker)" : "SSH";
  if (backend === "slurm") return "Slurm";
  return "HTTP API";
}

export function ClusterCard({ cluster, onRemove, onValidate, onResetEnv, onEdit }: ClusterCardProps) {
  const backend = cluster.backend || "slurm";
  const isValidated = !!cluster.validatedAt;
  const [validating, setValidating] = useState(false);
  const [validateResult, setValidateResult] = useState<"success" | "error" | null>(null);
  const [validateOutput, setValidateOutput] = useState("");
  const [resetting, setResetting] = useState(false);
  const [resetResult, setResetResult] = useState<"success" | "error" | null>(null);
  const [confirmRemove, setConfirmRemove] = useState(false);

  async function handleValidate() {
    setValidating(true);
    setValidateResult(null);
    setValidateOutput("");
    try {
      await onValidate((output) => setValidateOutput(output));
      setValidateResult("success");
    } catch {
      setValidateResult("error");
    } finally {
      setValidating(false);
      setTimeout(() => setValidateResult(null), 4000);
    }
  }

  async function handleResetEnv() {
    setResetting(true);
    setResetResult(null);
    try {
      await onResetEnv();
      setResetResult("success");
    } catch {
      setResetResult("error");
    } finally {
      setResetting(false);
      setTimeout(() => setResetResult(null), 4000);
    }
  }

  const accentColor = isValidated ? "var(--success)" : "var(--warning)";

  return (
    <div
      className="cluster-card"
      style={{ "--cluster-accent": accentColor } as React.CSSProperties}
    >
      {/* Header: name + status dot + actions */}
      <div className="cluster-card-header">
        <div className="flex-row">
          <span
            className={"job-status-dot"}
            style={{ background: accentColor }}
          />
          <span className="cluster-card-name">{cluster.name}</span>
          <span className="badge badge-muted">{getBackendLabel(backend, cluster.dockerImage)}</span>
          {isValidated ? (
            <span className="badge badge-success">
              <CheckCircle size={10} /> Validated
            </span>
          ) : backend !== "http-api" ? (
            <span className="badge badge-warning">
              <XCircle size={10} /> Not validated
            </span>
          ) : null}
        </div>
        <div className="flex-row">
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onEdit} title="Edit cluster">
            <Pencil size={12} />
          </button>
          {!confirmRemove ? (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={() => setConfirmRemove(true)} title="Remove cluster">
              <Trash2 size={12} />
            </button>
          ) : (
            <button
              className="btn btn-sm btn-error"
              onClick={() => { onRemove(); setConfirmRemove(false); }}
              onBlur={() => setConfirmRemove(false)}
              autoFocus
            >
              Confirm
            </button>
          )}
        </div>
      </div>

      {/* Connection details */}
      <div className="cluster-card-details">
        <div className="cluster-card-detail">
          <span className="cluster-card-detail-label">Host</span>
          <span className="cluster-card-detail-value">
            {cluster.user}@{cluster.host}{cluster.sshPort && cluster.sshPort !== 22 ? `:${cluster.sshPort}` : ""}
          </span>
        </div>
        {backend !== "http-api" && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">Workspace</span>
            <span className="cluster-card-detail-value">{cluster.remoteWorkspace}</span>
          </div>
        )}
        {backend !== "http-api" && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">Python</span>
            <span className="cluster-card-detail-value">{cluster.pythonPath}</span>
          </div>
        )}
        {backend === "slurm" && cluster.defaultPartition && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">Partition</span>
            <span className="cluster-card-detail-value">{cluster.defaultPartition}</span>
          </div>
        )}
        {backend === "ssh" && cluster.dockerImage && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">Docker Image</span>
            <span className="cluster-card-detail-value">{cluster.dockerImage}</span>
          </div>
        )}
        {backend === "http-api" && cluster.apiEndpoint && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">API Endpoint</span>
            <span className="cluster-card-detail-value">{cluster.apiEndpoint}</span>
          </div>
        )}
        {isValidated && cluster.validatedAt && (
          <div className="cluster-card-detail">
            <span className="cluster-card-detail-label">Validated</span>
            <span className="cluster-card-detail-value">{formatValidatedDate(cluster.validatedAt)}</span>
          </div>
        )}
      </div>
      {cluster.gpuTypes.length > 0 && (
        <div className="cluster-card-gpus">
          <span className="cluster-card-detail-label">GPUs</span>
          <div className="cluster-card-gpu-tags">
            {cluster.gpuTypes.map((gpu) => (
              <span key={gpu} className="cluster-gpu-tag" title={gpu}>
                {truncateGpuName(gpu)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons — validate for slurm + ssh, reset env for slurm only */}
      {backend !== "http-api" && (
        <div className="cluster-card-actions">
          <button className="btn btn-sm" onClick={handleValidate} disabled={validating}>
            {validating ? <Loader size={12} className="spin" /> : <RefreshCw size={12} />}
            {validating ? "Validating..." : validateResult === "success" ? "Validated!" : validateResult === "error" ? "Failed" : "Validate"}
          </button>
          {backend === "slurm" && (
            <button className="btn btn-sm" onClick={handleResetEnv} disabled={resetting}>
              {resetting ? <Loader size={12} className="spin" /> : <RotateCcw size={12} />}
              {resetting ? "Resetting..." : resetResult === "success" ? "Env Reset!" : resetResult === "error" ? "Reset Failed" : "Reset Env"}
            </button>
          )}
        </div>
      )}

      {/* Validation output */}
      {(validating || validateOutput) && (
        <pre className="console console-short">
          {validateOutput || "Connecting to cluster..."}
        </pre>
      )}
    </div>
  );
}
