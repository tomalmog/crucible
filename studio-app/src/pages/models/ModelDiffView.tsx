import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";

interface ModelDiffViewProps {
  dataRoot: string;
}

export function ModelDiffView({ dataRoot }: ModelDiffViewProps) {
  const command = useForgeCommand();
  const [name, setName] = useState("");
  const [versionA, setVersionA] = useState("");
  const [versionB, setVersionB] = useState("");

  async function runDiff() {
    if (!name.trim() || !versionA.trim() || !versionB.trim()) return;
    await command.run(dataRoot, [
      "model", "diff", "--name", name.trim(),
      "--version-a", versionA.trim(), "--version-b", versionB.trim(),
    ]);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Model Diff</h3>
      <div className="grid-3">
        <FormField label="Model Name">
          <input value={name} onChange={(e) => setName(e.currentTarget.value)} placeholder="my-model" />
        </FormField>
        <FormField label="Version A">
          <input value={versionA} onChange={(e) => setVersionA(e.currentTarget.value)} placeholder="v1" />
        </FormField>
        <FormField label="Version B">
          <input value={versionB} onChange={(e) => setVersionB(e.currentTarget.value)} placeholder="v2" />
        </FormField>
      </div>
      <button className="btn btn-primary gap-top" onClick={() => runDiff().catch(console.error)} disabled={command.isRunning}>
        Diff
      </button>
      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}
    </div>
  );
}
