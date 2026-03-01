import type { ModelVersion } from "../../types/models";

interface ModelOverviewProps {
  version: ModelVersion;
}

export function ModelOverview({ version }: ModelOverviewProps) {
  const fields = [
    { label: "Model Name", value: version.modelName },
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
      <table className="overview-table">
        <tbody>
          {fields.map((f) => (
            <tr key={f.label}>
              <td className="overview-label">{f.label}</td>
              <td className="overview-value">{f.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
