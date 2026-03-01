import { useEffect, useState } from "react";
import { getModelArchitecture } from "../../api/studioApi";
import type { ModelVersion } from "../../types/models";

interface ModelOverviewProps {
  version: ModelVersion;
}

const ARCH_FIELDS: { key: string; label: string }[] = [
  { key: "hidden_dim", label: "Embedding Dim" },
  { key: "num_layers", label: "Layers" },
  { key: "attention_heads", label: "Attention Heads" },
  { key: "mlp_hidden_dim", label: "MLP Hidden Dim" },
  { key: "mlp_layers", label: "MLP Layers" },
  { key: "dropout", label: "Dropout" },
  { key: "position_embedding_type", label: "Position Embedding" },
];

const TRAINING_FIELDS: { key: string; label: string }[] = [
  { key: "epochs", label: "Epochs" },
  { key: "learning_rate", label: "Learning Rate" },
  { key: "batch_size", label: "Batch Size" },
  { key: "optimizer_type", label: "Optimizer" },
  { key: "precision_mode", label: "Precision" },
  { key: "max_token_length", label: "Max Token Length" },
];

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
              {ARCH_FIELDS.map((f) =>
                config[f.key] != null ? (
                  <div key={f.key} className="metric-card">
                    <span className="metric-label">{f.label}</span>
                    <span className="metric-value">{String(config[f.key])}</span>
                  </div>
                ) : null,
              )}
            </div>
          </div>

          <div className="panel">
            <h3 className="panel-title">Training</h3>
            <div className="stats-grid">
              {TRAINING_FIELDS.map((f) =>
                config[f.key] != null ? (
                  <div key={f.key} className="metric-card">
                    <span className="metric-label">{f.label}</span>
                    <span className="metric-value">{String(config[f.key])}</span>
                  </div>
                ) : null,
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
