import { useState } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { PathInput } from "../../components/shared/PathInput";
import { SweepResultsView } from "./SweepResultsView";
import { Plus, Trash2 } from "lucide-react";

interface SweepParam {
  name: string;
  values: string;
}

const PARAM_NAME_OPTIONS = [
  "learning_rate", "batch_size", "hidden_dim", "num_layers",
  "attention_heads", "dropout", "weight_decay", "mlp_hidden_dim",
];

export function SweepConfigForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [dataset, setDataset] = useState("");
  const [outputDir, setOutputDir] = useState("./outputs/sweep");
  const [strategy, setStrategy] = useState("grid");
  const [maxTrials, setMaxTrials] = useState("10");
  const [metric, setMetric] = useState("validation_loss");
  const [maximize, setMaximize] = useState(false);
  const [params, setParams] = useState<SweepParam[]>([
    { name: "learning_rate", values: "0.001, 0.01, 0.1" },
  ]);
  const [results, setResults] = useState<string>("");

  function addParam() {
    setParams([...params, { name: "batch_size", values: "8, 16, 32" }]);
  }

  function removeParam(idx: number) {
    setParams(params.filter((_, i) => i !== idx));
  }

  function updateParam(idx: number, field: keyof SweepParam, value: string) {
    const updated = [...params];
    updated[idx] = { ...updated[idx], [field]: value };
    setParams(updated);
  }

  async function startSweep() {
    if (!dataRoot || !dataset.trim() || params.length === 0) return;
    const paramDefs = params.map((p) => ({
      name: p.name,
      values: p.values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v)),
    }));
    const paramsJson = JSON.stringify({ parameters: paramDefs });
    const args = [
      "sweep", "--dataset", dataset, "--output-dir", outputDir,
      "--params", paramsJson, "--strategy", strategy,
      "--max-trials", maxTrials, "--metric", metric, "--json",
    ];
    if (maximize) args.push("--maximize");
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && command.output) {
      setResults(command.output);
    }
  }

  if (results) {
    return <SweepResultsView output={results} onBack={() => setResults("")} />;
  }

  return (
    <div className="panel stack-lg">
      <h3>Hyperparameter Sweep</h3>
      <div className="grid-2">
        <FormField label="Dataset">
          <DatasetSelect value={dataset} onChange={setDataset} />
        </FormField>
        <FormField label="Output Directory">
          <PathInput value={outputDir} onChange={setOutputDir} kind="folder" />
        </FormField>
      </div>

      <div className="stack-sm">
        <div className="row-between">
          <span className="text-sm text-secondary">Parameters</span>
          <button className="btn btn-ghost btn-sm" onClick={addParam}>
            <Plus size={13} /> Add
          </button>
        </div>
        {params.map((p, i) => (
          <div className="row" key={i} style={{ alignItems: "flex-end" }}>
            <div className="grid-2" style={{ flex: 1 }}>
              <FormField label="Name">
                <select value={p.name} onChange={(e) => updateParam(i, "name", e.currentTarget.value)}>
                  {PARAM_NAME_OPTIONS.map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </FormField>
              <FormField label="Values (comma-separated)">
                <input value={p.values} onChange={(e) => updateParam(i, "values", e.currentTarget.value)} />
              </FormField>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => removeParam(i)} disabled={params.length <= 1} style={{ marginBottom: 2 }}>
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <FormField label="Strategy">
          <select value={strategy} onChange={(e) => setStrategy(e.currentTarget.value)}>
            <option value="grid">Grid Search</option>
            <option value="random">Random Search</option>
          </select>
        </FormField>
        <FormField label="Max Trials">
          <input type="number" value={maxTrials} onChange={(e) => setMaxTrials(e.currentTarget.value)} disabled={strategy !== "random"} />
        </FormField>
        <FormField label="Metric">
          <input value={metric} onChange={(e) => setMetric(e.currentTarget.value)} />
        </FormField>
        <FormField label="Direction">
          <select value={maximize ? "maximize" : "minimize"} onChange={(e) => setMaximize(e.currentTarget.value === "maximize")}>
            <option value="minimize">Minimize</option>
            <option value="maximize">Maximize</option>
          </select>
        </FormField>
      </div>

      <button
        className="btn btn-primary"
        onClick={() => startSweep().catch(console.error)}
        disabled={command.isRunning || !dataset.trim() || params.length === 0}
      >
        {command.isRunning ? "Running Sweep..." : "Start Sweep"}
      </button>
      {command.error && <p className="error-text">{command.error}</p>}
    </div>
  );
}
