import { useState, useMemo } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";

export function EvalResultsView() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("");
  const [benchmarks, setBenchmarks] = useState("mmlu,gsm8k,hellaswag,arc");
  const [results, setResults] = useState<string>("");

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model path");
    return m;
  }, [modelPath]);

  async function runEval() {
    if (!dataRoot) return;
    const args = ["eval", "--model-path", modelPath, "--benchmarks", benchmarks];
    if (baseModelPath.trim()) {
      args.push("--base-model", baseModelPath);
    }
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
  }

  return (
    <CommandFormPanel
      title="Model Evaluation"
      missing={missing}
      isRunning={command.isRunning}
      submitLabel="Run Evaluation"
      runningLabel="Evaluating..."
      onSubmit={() => runEval().catch(console.error)}
      error={command.error}
      output={results}
    >
      <div className="grid-2">
        <FormField label="Model Path" required>
          <PathInput value={modelPath} onChange={setModelPath} placeholder="/path/to/model.pt" filters={[{ name: "Model", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Base Model" hint="optional, for comparison">
          <PathInput value={baseModelPath} onChange={setBaseModelPath} placeholder="/path/to/base_model.pt" filters={[{ name: "Model", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Benchmarks" hint="comma-separated">
          <input value={benchmarks} onChange={(e) => setBenchmarks(e.currentTarget.value)} />
        </FormField>
      </div>
    </CommandFormPanel>
  );
}
