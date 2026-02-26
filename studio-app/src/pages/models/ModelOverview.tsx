import type { ModelVersion } from "../../types/models";

interface ModelOverviewProps {
  version: ModelVersion;
}

export function ModelOverview({ version }: ModelOverviewProps) {
  const fields = [
    { label: "Version ID", value: version.versionId },
    { label: "Model Path", value: version.modelPath },
    { label: "Training Run", value: version.runId ?? "\u2014" },
    { label: "Parent Version", value: version.parentVersionId ?? "\u2014" },
    { label: "Created", value: version.createdAt || "\u2014" },
    { label: "Status", value: version.isActive ? "Active" : "Inactive" },
  ];

  return (
    <div className="panel">
      <h3 className="panel-title">Version Overview</h3>
      <div className="stats-grid">
        {fields.map((f) => (
          <div key={f.label} className="metric-card">
            <span className="metric-label">{f.label}</span>
            <span className="metric-value">{f.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
