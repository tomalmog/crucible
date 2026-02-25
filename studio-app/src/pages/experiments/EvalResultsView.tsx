import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function EvalResultsView() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [modelPath, setModelPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("");
  const [benchmarks, setBenchmarks] = useState("mmlu,gsm8k,hellaswag,arc");
  const [results, setResults] = useState<string>("");
  const [running, setRunning] = useState(false);

  async function runEval() {
    if (!dataRoot || !modelPath.trim()) return;
    setRunning(true);
    const args = ["eval", "--model-path", modelPath, "--benchmarks", benchmarks];
    if (baseModelPath.trim()) {
      args.push("--base-model", baseModelPath);
    }
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
    setRunning(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Model Evaluation</h3>
      <div className="grid-2">
        <FormField label="Model Path">
          <input value={modelPath} onChange={(e) => setModelPath(e.currentTarget.value)} placeholder="/path/to/model.pt" />
        </FormField>
        <FormField label="Base Model (optional, for comparison)">
          <input value={baseModelPath} onChange={(e) => setBaseModelPath(e.currentTarget.value)} placeholder="/path/to/base_model.pt" />
        </FormField>
        <FormField label="Benchmarks (comma-separated)">
          <input value={benchmarks} onChange={(e) => setBenchmarks(e.currentTarget.value)} />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => runEval().catch(console.error)} disabled={running || !modelPath.trim()}>
        {running ? "Evaluating..." : "Run Evaluation"}
      </button>
      {results && <pre className="console">{results}</pre>}
    </div>
  );
}
