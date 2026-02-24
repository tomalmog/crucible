import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { buildSafetyEvalArgs } from "../../api/commandArgs";
import { FormField } from "../../components/shared/FormField";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface SafetyEvalFormProps {
  dataRoot: string;
  onResult: (output: string) => void;
}

export function SafetyEvalForm({ dataRoot, onResult }: SafetyEvalFormProps) {
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [dataset, setDataset] = useState("");
  const [categories, setCategories] = useState("");
  const [threshold, setThreshold] = useState("0.5");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const extra: Record<string, string> = {};
    if (modelPath.trim()) extra["--model-path"] = modelPath.trim();
    if (dataset.trim()) extra["--dataset"] = dataset.trim();
    if (categories.trim()) extra["--categories"] = categories.trim();
    if (threshold.trim()) extra["--threshold"] = threshold.trim();
    const status = await command.run(dataRoot, buildSafetyEvalArgs(extra));
    onResult(status.stdout || status.stderr);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Safety Evaluation</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <div className="grid-2">
          <FormField label="Model Path">
            <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
          </FormField>
          <FormField label="Dataset">
            <input value={dataset} onChange={(e) => setDataset(e.currentTarget.value)} placeholder="safety-eval-set" />
          </FormField>
          <FormField label="Categories (comma-separated)">
            <input value={categories} onChange={(e) => setCategories(e.currentTarget.value)} placeholder="toxicity,bias,fairness" />
          </FormField>
          <FormField label="Threshold">
            <input value={threshold} onChange={(e) => setThreshold(e.currentTarget.value)} />
          </FormField>
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Evaluating..." : "Run Evaluation"}
        </button>
      </form>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Safety eval..." percent={command.status.progress_percent} />
        </div>
      )}
    </div>
  );
}
