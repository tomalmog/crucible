import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { FormField } from "../../components/shared/FormField";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { CommandProgress } from "../../components/shared/CommandProgress";

interface LatencyProfileFormProps {
  dataRoot: string;
}

export function LatencyProfileForm({ dataRoot }: LatencyProfileFormProps) {
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [batchSizes, setBatchSizes] = useState("1,4,8");
  const [warmupRuns, setWarmupRuns] = useState("3");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    const args = ["deploy", "profile"];
    if (modelPath.trim()) args.push("--model-path", modelPath.trim());
    if (batchSizes.trim()) args.push("--batch-sizes", batchSizes.trim());
    if (warmupRuns.trim()) args.push("--warmup-runs", warmupRuns.trim());
    await command.run(dataRoot, args);
  }

  return (
    <div className="panel">
      <h3 className="panel-title">Latency Profile</h3>
      <form onSubmit={(e) => onSubmit(e).catch(console.error)} className="stack">
        <div className="grid-3">
          <FormField label="Model Path">
            <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
          </FormField>
          <FormField label="Batch Sizes">
            <input value={batchSizes} onChange={(e) => setBatchSizes(e.currentTarget.value)} placeholder="1,4,8" />
          </FormField>
          <FormField label="Warmup Runs">
            <input type="number" value={warmupRuns} onChange={(e) => setWarmupRuns(e.currentTarget.value)} />
          </FormField>
        </div>
        <button className="btn btn-primary btn-lg" type="submit" disabled={command.isRunning}>
          {command.isRunning ? "Profiling..." : "Run Profile"}
        </button>
      </form>
      {command.isRunning && command.status && (
        <div className="gap-top">
          <CommandProgress label="Profiling..." percent={command.status.progress_percent} />
        </div>
      )}
      {command.output && <div className="gap-top"><StatusConsole output={command.output} /></div>}
    </div>
  );
}
