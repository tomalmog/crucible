import { useCallback, useEffect, useRef, useState } from "react";
import { getModelArchitecture, startForgeCommand, getForgeCommandStatus } from "../../api/studioApi";
import { useForge } from "../../context/ForgeContext";
import type { ModelVersion } from "../../types/models";
import { Download, CheckCircle, Loader } from "lucide-react";

interface ModelOverviewProps {
  version: ModelVersion;
}

// Each entry can have multiple keys (Forge name, HuggingFace name) — first match wins
const ARCH_FIELDS: { keys: string[]; label: string }[] = [
  { keys: ["architectures"], label: "Architecture" },
  { keys: ["model_type"], label: "Model Type" },
  { keys: ["hidden_dim", "hidden_size"], label: "Hidden Size" },
  { keys: ["num_layers", "num_hidden_layers"], label: "Layers" },
  { keys: ["attention_heads", "num_attention_heads"], label: "Attention Heads" },
  { keys: ["mlp_hidden_dim", "intermediate_size"], label: "Intermediate Size" },
  { keys: ["vocab_size"], label: "Vocab Size" },
  { keys: ["dropout", "hidden_dropout_prob"], label: "Dropout" },
  { keys: ["position_embedding_type"], label: "Position Embedding" },
  { keys: ["torch_dtype"], label: "Dtype" },
];

const TRAINING_FIELDS: { keys: string[]; label: string }[] = [
  { keys: ["epochs"], label: "Epochs" },
  { keys: ["learning_rate"], label: "Learning Rate" },
  { keys: ["batch_size"], label: "Batch Size" },
  { keys: ["optimizer_type"], label: "Optimizer" },
  { keys: ["precision_mode"], label: "Precision" },
  { keys: ["max_token_length", "max_position_embeddings"], label: "Max Token Length" },
];

function resolveField(config: Record<string, unknown>, keys: string[]): unknown | undefined {
  for (const k of keys) {
    if (config[k] != null) {
      const v = config[k];
      return Array.isArray(v) ? v.join(", ") : v;
    }
  }
  return undefined;
}

export function ModelOverview({ version }: ModelOverviewProps) {
  const { dataRoot } = useForge();
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [pulling, setPulling] = useState(false);
  const [pullProgress, setPullProgress] = useState<string[]>([]);
  const [pullDone, setPullDone] = useState(false);
  const [pullError, setPullError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelArchitecture(version.modelPath)
      .then((data) => { if (!cancelled) setConfig(data); })
      .catch(() => { if (!cancelled) setConfig(null); });
    return () => { cancelled = true; };
  }, [version.modelPath]);

  const handlePull = useCallback(async () => {
    if (!dataRoot || !version.runId) return;
    setPulling(true);
    setPullProgress([]);
    setPullError(null);
    setPullDone(false);
    try {
      const task = await startForgeCommand(dataRoot, [
        "remote", "pull-model",
        "--job-id", version.runId,
        "--model-name", version.modelName,
      ]);
      // Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const status = await getForgeCommandStatus(task.task_id);
          // Extract progress lines from stdout
          const lines = (status.stdout || "")
            .split("\n")
            .filter((l: string) => l.startsWith("FORGE_PULL_PROGRESS: "))
            .map((l: string) => l.replace("FORGE_PULL_PROGRESS: ", ""));
          if (lines.length > 0) setPullProgress(lines);
          if (status.status !== "running") {
            if (pollRef.current) clearInterval(pollRef.current);
            if (status.status === "completed") {
              setPullDone(true);
            } else {
              setPullError(status.stderr || "Pull failed");
            }
            setPulling(false);
          }
        } catch {
          if (pollRef.current) clearInterval(pollRef.current);
          setPulling(false);
          setPullError("Lost connection to pull task");
        }
      }, 2000);
    } catch (err) {
      setPulling(false);
      setPullError(`Failed to start pull: ${err}`);
    }
  }, [dataRoot, version.runId, version.modelName]);

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const isRemoteOnly = version.locationType === "remote";
  const locationLabel = version.locationType === "both" ? "Local + Remote"
    : version.locationType === "remote" ? "Remote" : "Local";

  const infoFields = [
    { label: "Model Name", value: version.modelName },
    { label: "Version ID", value: version.versionId },
    { label: "Location", value: locationLabel },
    { label: "Model Path", value: version.modelPath || "\u2014" },
    ...(version.remoteHost ? [{ label: "Remote", value: `${version.remoteHost}:${version.remotePath}` }] : []),
    { label: "Training Run", value: version.runId ?? "\u2014" },
    { label: "Parent Version", value: version.parentVersionId ?? "\u2014" },
    { label: "Created", value: version.createdAt || "\u2014" },
    { label: "Status", value: version.isActive ? "Active" : "Inactive" },
  ];

  return (
    <div className="stack-lg">
      <div className="panel">
        <h3 className="panel-title">Version Info</h3>
        <table className="overview-table">
          <tbody>
            {infoFields.map((f) => (
              <tr key={f.label}>
                <td className="overview-label">{f.label}</td>
                <td className="overview-value">{f.value}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {isRemoteOnly && version.runId && (
          <div className="gap-top-md">
            {!pulling && !pullDone && (
              <button className="btn btn-primary" onClick={handlePull}>
                <Download size={14} /> Pull to Local
              </button>
            )}
            {pulling && (
              <div className="pull-progress">
                <div className="flex-row">
                  <Loader size={14} className="spin" />
                  <span>Pulling model from cluster...</span>
                </div>
                {pullProgress.length > 0 && (
                  <div className="pull-steps">
                    {pullProgress.map((step, i) => (
                      <div key={i} className="pull-step">{step}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {pullDone && (
              <div className="pull-success flex-row">
                <CheckCircle size={14} />
                <span>Model pulled successfully!</span>
              </div>
            )}
            {pullError && (
              <div className="error-alert-prominent">{pullError}</div>
            )}
          </div>
        )}
      </div>

      {config && (
        <>
          <div className="panel">
            <h3 className="panel-title">Architecture</h3>
            <div className="stats-grid">
              {ARCH_FIELDS.map((f) => {
                const v = resolveField(config, f.keys);
                return v !== undefined ? (
                  <div key={f.label} className="metric-card">
                    <span className="metric-label">{f.label}</span>
                    <span className="metric-value">{String(v)}</span>
                  </div>
                ) : null;
              })}
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title">Training</h3>
            <div className="stats-grid">
              {TRAINING_FIELDS.map((f) => {
                const v = resolveField(config, f.keys);
                return v !== undefined ? (
                  <div key={f.label} className="metric-card">
                    <span className="metric-label">{f.label}</span>
                    <span className="metric-value">{String(v)}</span>
                  </div>
                ) : null;
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
