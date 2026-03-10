import { useState } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import type { ModelVersion } from "../../types/models";

interface ModelDiffViewProps {
  dataRoot: string;
  versions: ModelVersion[];
}

export function ModelDiffView({ dataRoot, versions }: ModelDiffViewProps) {
  const command = useCrucibleCommand();
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");

  async function runDiff() {
    if (!versionA || !versionB) return;
    const va = versions.find((v) => v.versionId === versionA);
    const vb = versions.find((v) => v.versionId === versionB);
    if (!va || !vb) return;
    await command.run(dataRoot, [
      "model", "diff",
      "--version-a", versionA,
      "--version-b", versionB,
    ]);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Model Diff</h3>
      <div className="grid-2">
        <FormField label="Version A">
          <select value={versionA} onChange={(e) => setVersionA(e.currentTarget.value)}>
            <option value="">Select version...</option>
            {versions.map((v) => (
              <option key={v.versionId} value={v.versionId}>
                {v.versionId.slice(0, 16)}... {v.isActive ? "(active)" : ""}
              </option>
            ))}
          </select>
        </FormField>
        <FormField label="Version B">
          <select value={versionB} onChange={(e) => setVersionB(e.currentTarget.value)}>
            <option value="">Select version...</option>
            {versions.map((v) => (
              <option key={v.versionId} value={v.versionId}>
                {v.versionId.slice(0, 16)}... {v.isActive ? "(active)" : ""}
              </option>
            ))}
          </select>
        </FormField>
      </div>
      <button className="btn btn-primary gap-top" onClick={() => runDiff().catch(console.error)} disabled={command.isRunning || !versionA || !versionB}>
        {command.isRunning ? "Running..." : "Diff"}
      </button>
      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}
    </div>
  );
}
