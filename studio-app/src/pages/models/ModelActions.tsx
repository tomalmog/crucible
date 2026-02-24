import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";

interface ModelActionsProps {
  dataRoot: string;
}

export function ModelActions({ dataRoot }: ModelActionsProps) {
  const command = useForgeCommand();
  const [action, setAction] = useState<"tag" | "rollback" | "export">("tag");
  const [name, setName] = useState("");
  const [version, setVersion] = useState("");
  const [tag, setTag] = useState("");
  const [exportPath, setExportPath] = useState("");

  async function runAction() {
    if (!name.trim() || !version.trim()) return;
    if (action === "tag") {
      await command.run(dataRoot, ["model", "tag", "--name", name.trim(), "--version", version.trim(), "--tag", tag.trim()]);
    } else if (action === "rollback") {
      await command.run(dataRoot, ["model", "rollback", "--name", name.trim(), "--version", version.trim()]);
    } else {
      const args = ["export-spec", "--name", name.trim(), "--version", version.trim()];
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

      <div className="grid-2">
        <FormField label="Model Name">
          <input value={name} onChange={(e) => setName(e.currentTarget.value)} placeholder="my-model" />
        </FormField>
        <FormField label="Version">
          <input value={version} onChange={(e) => setVersion(e.currentTarget.value)} placeholder="v1" />
        </FormField>
        {action === "tag" && (
          <FormField label="Tag">
            <input value={tag} onChange={(e) => setTag(e.currentTarget.value)} placeholder="production" />
          </FormField>
        )}
        {action === "export" && (
          <FormField label="Export Path (optional)">
            <input value={exportPath} onChange={(e) => setExportPath(e.currentTarget.value)} placeholder="./exports" />
          </FormField>
        )}
      </div>

      <button className="btn btn-primary gap-top" onClick={() => runAction().catch(console.error)} disabled={command.isRunning}>
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
