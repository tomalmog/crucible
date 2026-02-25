import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function HubPushForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [repoId, setRepoId] = useState("");
  const [message, setMessage] = useState("Upload model via Forge");
  const [isPrivate, setIsPrivate] = useState(false);
  const [pushing, setPushing] = useState(false);
  const [result, setResult] = useState("");

  async function pushModel() {
    if (!dataRoot || !modelPath.trim() || !repoId.trim()) return;
    setPushing(true);
    const args = ["hub", "push", modelPath, repoId, "--message", message];
    if (isPrivate) args.push("--private");
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResult(command.output);
    }
    setPushing(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Push Model to Hub</h3>
      <div className="grid-2">
        <FormField label="Model Path">
          <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model" />
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
      <button className="btn btn-primary" onClick={() => pushModel().catch(console.error)} disabled={pushing || !modelPath.trim() || !repoId.trim()}>
        {pushing ? "Pushing..." : "Push to Hub"}
      </button>
      {result && <pre className="console">{result}</pre>}
    </div>
  );
}
