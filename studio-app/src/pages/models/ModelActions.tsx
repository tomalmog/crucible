import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { PathInput } from "../../components/shared/PathInput";
import type { ModelVersion } from "../../types/models";

interface ModelActionsProps {
  dataRoot: string;
  versions: ModelVersion[];
}

export function ModelActions({ dataRoot, versions }: ModelActionsProps) {
  const command = useForgeCommand();
  const [action, setAction] = useState<"tag" | "rollback" | "export">("tag");
  const [version, setVersion] = useState("");
  const [tag, setTag] = useState("");
  const [exportPath, setExportPath] = useState("");

  async function runAction() {
    if (!version) return;
    const v = versions.find((mv) => mv.versionId === version);
    if (!v) return;
    if (action === "tag") {
      if (!tag.trim()) return;
      await command.run(dataRoot, ["model", "tag", "--name", v.modelPath, "--version", version, "--tag", tag.trim()]);
    } else if (action === "rollback") {
      await command.run(dataRoot, ["model", "rollback", "--name", v.modelPath, "--version", version]);
    } else {
      const args = ["export-spec", "--name", v.modelPath, "--version", version];
      if (exportPath.trim()) args.push("--output", exportPath.trim());
      await command.run(dataRoot, args);
    }
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Model Actions</h3>
      <div className="action-group">
        {(["tag", "rollback", "export"] as const).map((a) => (
          <button key={a} className={`btn btn-sm ${action === a ? "btn-primary" : ""}`} onClick={() => setAction(a)}>
            {a.charAt(0).toUpperCase() + a.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid-2 gap-top">
        <FormField label="Version">
          <select value={version} onChange={(e) => setVersion(e.currentTarget.value)}>
            <option value="">Select version...</option>
            {versions.map((v) => (
              <option key={v.versionId} value={v.versionId}>
                {v.versionId.slice(0, 16)}... {v.isActive ? "(active)" : ""}
              </option>
            ))}
          </select>
        </FormField>
        {action === "tag" && (
          <FormField label="Tag">
            <input value={tag} onChange={(e) => setTag(e.currentTarget.value)} placeholder="production" />
          </FormField>
        )}
        {action === "export" && (
          <FormField label="Export Path (optional)">
            <PathInput value={exportPath} onChange={setExportPath} placeholder="./exports" kind="folder" />
          </FormField>
        )}
      </div>

      <button className="btn btn-primary gap-top" onClick={() => runAction().catch(console.error)} disabled={command.isRunning || !version}>
        {command.isRunning ? "Running..." : `Run ${action}`}
      </button>

      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}
    </div>
  );
}
