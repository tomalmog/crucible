import { useState } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { PathInput } from "../../components/shared/PathInput";
import type { ModelVersion } from "../../types/models";

interface ModelActionsProps {
  dataRoot: string;
  versions: ModelVersion[];
  modelName: string | null;
}

type DeleteScope = "registry" | "local" | "remote" | "both";

export function ModelActions({ dataRoot, versions, modelName }: ModelActionsProps) {
  const command = useCrucibleCommand();
  const { refreshModels } = useCrucible();
  const [action, setAction] = useState<"tag" | "rollback" | "export" | "delete">("tag");
  const [version, setVersion] = useState("");
  const [tag, setTag] = useState("");
  const [exportPath, setExportPath] = useState("");
  const [deleteScope, setDeleteScope] = useState<DeleteScope>("local");
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  async function runAction() {
    if (action === "delete") return;
    if (!version) return;
    if (action === "tag") {
      if (!tag.trim()) return;
      await command.run(dataRoot, ["model", "tag", "--version-id", version, "--tag", tag.trim()]);
    } else if (action === "rollback") {
      if (!modelName) return;
      await command.run(dataRoot, ["model", "rollback", "--name", modelName, "--version-id", version]);
    } else {
      const args = ["export-spec", "--run-id", version];
      if (exportPath.trim()) args.push("--output", exportPath.trim());
      await command.run(dataRoot, args);
    }
  }

  async function runDelete() {
    if (!modelName) return;
    const args = ["model", "delete", "--name", modelName, "--yes"];
    if (version) args.push("--version-id", version);
    if (deleteScope === "local" || deleteScope === "both") args.push("--delete-local");
    if (deleteScope === "remote" || deleteScope === "both") args.push("--include-remote");
    if (deleteScope === "remote") args.push("--keep-registry");
    await command.run(dataRoot, args);
    setConfirmingDelete(false);
    await refreshModels();
  }

  const deletePathsList = versions
    .filter((v) => (version ? v.versionId === version : true))
    .map((v) => v.modelPath)
    .filter(Boolean);

  const hasRemote = versions.some((v) => v.remotePath);

  return (
    <div className="panel">
      <h3 className="panel-title">Model Actions</h3>
      <div className="action-group">
        {(["tag", "rollback", "export", "delete"] as const).map((a) => (
          <button
            key={a}
            className={`btn btn-sm ${action === a ? "btn-primary" : ""}`}
            onClick={() => { setAction(a); setConfirmingDelete(false); }}
          >
            {a.charAt(0).toUpperCase() + a.slice(1)}
          </button>
        ))}
      </div>

      <div className="grid-2 gap-top">
        {action !== "delete" && (
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
        )}
        {action === "delete" && (
          <>
            <FormField label="Version (optional)">
              <select value={version} onChange={(e) => { setVersion(e.currentTarget.value); setConfirmingDelete(false); }}>
                <option value="">All versions (entire model)</option>
                {versions.map((v) => (
                  <option key={v.versionId} value={v.versionId}>
                    {v.versionId.slice(0, 16)}... {v.isActive ? "(active)" : ""}
                  </option>
                ))}
              </select>
            </FormField>
            <FormField label="Delete from">
              <select value={deleteScope} onChange={(e) => { setDeleteScope(e.currentTarget.value as DeleteScope); setConfirmingDelete(false); }}>
                <option value="registry">Registry only (keep files)</option>
                <option value="local">Local files</option>
                {hasRemote && <option value="remote">Remote files</option>}
                {hasRemote && <option value="both">Local + Remote</option>}
              </select>
            </FormField>
          </>
        )}
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

      {action === "delete" && deletePathsList.length > 0 && (deleteScope === "local" || deleteScope === "both") && (
        <div className="gap-top" style={{ color: "var(--error)", fontSize: "0.85em" }}>
          <strong>Local files to delete:</strong>
          {deletePathsList.map((p) => (
            <div key={p} style={{ marginLeft: 12 }}>{p}</div>
          ))}
        </div>
      )}

      {action !== "delete" && (
        <button
          className="btn btn-primary gap-top"
          onClick={() => runAction().catch(console.error)}
          disabled={command.isRunning || !version}
        >
          {command.isRunning ? "Running..." : `Run ${action}`}
        </button>
      )}

      {action === "delete" && !confirmingDelete && (
        <button
          className="btn btn-error gap-top"
          onClick={() => setConfirmingDelete(true)}
          disabled={command.isRunning || !modelName}
        >
          Delete
        </button>
      )}

      {action === "delete" && confirmingDelete && (
        <div className="gap-top" style={{ display: "flex", gap: 8 }}>
          <button
            className="btn btn-error"
            onClick={() => runDelete().catch(console.error)}
            disabled={command.isRunning}
          >
            {command.isRunning ? "Deleting..." : "Confirm Delete"}
          </button>
          <button className="btn" onClick={() => setConfirmingDelete(false)} disabled={command.isRunning}>
            Cancel
          </button>
        </div>
      )}

      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}
    </div>
  );
}
