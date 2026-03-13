import { useState } from "react";
import { CheckCircle, XCircle, Trash2, RefreshCw, RotateCcw, Loader, Pencil } from "lucide-react";
import type { ClusterConfig } from "../../types/remote";

interface ClusterCardProps {
  cluster: ClusterConfig;
  onRemove: () => void;
  onValidate: (onProgress: (output: string) => void) => Promise<void>;
  onResetEnv: () => Promise<void>;
  onEdit: () => void;
}

export function ClusterCard({ cluster, onRemove, onValidate, onResetEnv, onEdit }: ClusterCardProps) {
  const isValidated = !!cluster.validatedAt;
  const [validating, setValidating] = useState(false);
  const [validateResult, setValidateResult] = useState<"success" | "error" | null>(null);
  const [validateOutput, setValidateOutput] = useState("");
  const [resetting, setResetting] = useState(false);
  const [resetResult, setResetResult] = useState<"success" | "error" | null>(null);

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
          <button className="btn btn-sm" onClick={handleValidate} disabled={validating} title="Validate cluster">
            {validating ? <Loader size={12} className="spin" /> : <RefreshCw size={12} />}
            {validating ? "Validating..." : validateResult === "success" ? "Validated!" : validateResult === "error" ? "Failed" : "Validate"}
          </button>
          <button className="btn btn-sm" onClick={handleResetEnv} disabled={resetting} title="Remove and rebuild crucible conda env on next job">
            {resetting ? <Loader size={12} className="spin" /> : <RotateCcw size={12} />}
            {resetting ? "Resetting..." : resetResult === "success" ? "Env Reset!" : resetResult === "error" ? "Reset Failed" : "Reset Env"}
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onEdit} title="Edit cluster">
            <Pencil size={12} />
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

      {(validating || validateOutput) && (
        <pre className="console" style={{ maxHeight: 200, overflow: "auto" }}>
          {validateOutput || "Connecting to cluster..."}
        </pre>
      )}
    </div>
  );
}
