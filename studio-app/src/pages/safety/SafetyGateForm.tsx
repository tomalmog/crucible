import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { buildSafetyGateArgs } from "../../api/commandArgs";
import { FormField } from "../../components/shared/FormField";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface SafetyGateFormProps {
  dataRoot: string;
  onResult: (output: string) => void;
}

export function SafetyGateForm({ dataRoot, onResult }: SafetyGateFormProps) {
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [minScore, setMinScore] = useState("0.8");
  const [maxToxicity, setMaxToxicity] = useState("0.1");
  const [requireCategories, setRequireCategories] = useState("");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const extra: Record<string, string> = {};
    if (modelPath.trim()) extra["--model-path"] = modelPath.trim();
    if (minScore.trim()) extra["--min-score"] = minScore.trim();
    if (maxToxicity.trim()) extra["--max-toxicity"] = maxToxicity.trim();
    if (requireCategories.trim()) extra["--require-categories"] = requireCategories.trim();
    const status = await command.run(dataRoot, buildSafetyGateArgs(extra));
    onResult(status.stdout || status.stderr);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Deployment Gate</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <div className="grid-2">
          <FormField label="Model Path">
            <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
          </FormField>
          <FormField label="Min Score">
            <input value={minScore} onChange={(e) => setMinScore(e.currentTarget.value)} />
          </FormField>
          <FormField label="Max Toxicity">
            <input value={maxToxicity} onChange={(e) => setMaxToxicity(e.currentTarget.value)} />
          </FormField>
          <FormField label="Required Categories">
            <input value={requireCategories} onChange={(e) => setRequireCategories(e.currentTarget.value)} placeholder="all" />
          </FormField>
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Checking..." : "Run Gate Check"}
        </button>
      </form>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Gate check..." percent={command.status.progress_percent} />
        </div>
      )}
    </div>
  );
}
