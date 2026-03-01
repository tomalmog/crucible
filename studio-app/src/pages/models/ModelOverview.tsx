import { useEffect, useState } from "react";
import { getModelArchitecture } from "../../api/studioApi";
import type { ModelVersion } from "../../types/models";

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
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelArchitecture(version.modelPath)
      .then((data) => { if (!cancelled) setConfig(data); })
      .catch(() => { if (!cancelled) setConfig(null); });
    return () => { cancelled = true; };
  }, [version.modelPath]);

  const infoFields = [
    { label: "Model Name", value: version.modelName },
    { label: "Version ID", value: version.versionId },
    { label: "Model Path", value: version.modelPath },
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
