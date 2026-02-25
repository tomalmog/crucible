import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";

export function SweepConfigForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [dataset, setDataset] = useState("");
  const [outputDir, setOutputDir] = useState("./outputs/sweep");
  const [lrMin, setLrMin] = useState("1e-5");
  const [lrMax, setLrMax] = useState("1e-2");
  const [batchSizes, setBatchSizes] = useState("8,16,32");
  const [loraRanks, setLoraRanks] = useState("4,8,16");
  const [searchType, setSearchType] = useState("grid");
  const [maxTrials, setMaxTrials] = useState("10");
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState("");

  async function startSweep() {
    if (!dataRoot || !dataset.trim()) return;
    setRunning(true);
    const args = [
      "sweep",
      "--dataset", dataset,
      "--output-dir", outputDir,
      "--lr-min", lrMin,
      "--lr-max", lrMax,
      "--batch-sizes", batchSizes,
      "--max-trials", maxTrials,
    ];
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
    setRunning(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Hyperparameter Sweep</h3>
      <div className="grid-2">
        <FormField label="Dataset">
          <input value={dataset} onChange={(e) => setDataset(e.currentTarget.value)} placeholder="my-dataset" />
        </FormField>
        <FormField label="Output Directory">
          <input value={outputDir} onChange={(e) => setOutputDir(e.currentTarget.value)} />
        </FormField>
        <FormField label="Learning Rate Min">
          <input value={lrMin} onChange={(e) => setLrMin(e.currentTarget.value)} />
        </FormField>
        <FormField label="Learning Rate Max">
          <input value={lrMax} onChange={(e) => setLrMax(e.currentTarget.value)} />
        </FormField>
        <FormField label="Batch Sizes (comma-separated)">
          <input value={batchSizes} onChange={(e) => setBatchSizes(e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Ranks (comma-separated)">
          <input value={loraRanks} onChange={(e) => setLoraRanks(e.currentTarget.value)} />
        </FormField>
        <FormField label="Search Type">
          <select value={searchType} onChange={(e) => setSearchType(e.currentTarget.value)}>
            <option value="grid">Grid Search</option>
            <option value="random">Random Search</option>
          </select>
        </FormField>
        <FormField label="Max Trials">
          <input value={maxTrials} onChange={(e) => setMaxTrials(e.currentTarget.value)} />
        </FormField>
      </div>
      <button className="btn btn-primary" onClick={() => startSweep().catch(console.error)} disabled={running || !dataset.trim()}>
        {running ? "Running Sweep..." : "Start Sweep"}
      </button>
      {results && <pre className="console">{results}</pre>}
    </div>
  );
}
