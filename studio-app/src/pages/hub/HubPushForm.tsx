import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";
import { StatusConsole } from "../../components/shared/StatusConsole";

export function HubPushForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [repoId, setRepoId] = useState("");
  const [message, setMessage] = useState("Upload model via Forge");
  const [isPrivate, setIsPrivate] = useState(false);

  async function pushModel() {
    if (!dataRoot || !modelPath.trim() || !repoId.trim()) return;
    const args = ["hub", "push", modelPath, repoId, "--message", message];
    if (isPrivate) args.push("--private");
    await command.run(dataRoot, args);
  }

  return (
    <div className="panel stack-md">
      <h3 className="panel-title">Push Model to Hub</h3>
      <div className="grid-2">
        <FormField label="Model Path">
          <PathInput value={modelPath} onChange={setModelPath} placeholder="/path/to/model" kind="folder" />
        </FormField>
        <FormField label="Repository ID">
          <input value={repoId} onChange={(e) => setRepoId(e.currentTarget.value)} placeholder="username/model-name" />
        </FormField>
        <FormField label="Commit Message">
          <input value={message} onChange={(e) => setMessage(e.currentTarget.value)} />
        </FormField>
        <FormField label="Private Repository">
          <label className="checkbox-label">
            <input type="checkbox" checked={isPrivate} onChange={(e) => setIsPrivate(e.currentTarget.checked)} />
            Make repository private
          </label>
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => pushModel().catch(console.error)} disabled={command.isRunning || !modelPath.trim() || !repoId.trim()}>
        {command.isRunning ? "Pushing..." : "Push to Hub"}
      </button>
      {command.output && (
        <div className="gap-top">
          <StatusConsole output={command.output} />
        </div>
      )}
    </div>
  );
}
