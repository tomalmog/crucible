import { useState, useMemo } from "react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { PathInput } from "../../components/shared/PathInput";
import { MetricSelect } from "../../components/shared/MetricSelect";
import { SweepResultsView } from "./SweepResultsView";
import {
  TrainingMethod,
  TRAINING_METHODS,
  REQUIRED_METHOD_FIELDS,
} from "../../types/training";
import { Plus, Trash2, Check } from "lucide-react";

interface SweepParam {
  name: string;
  values: string;
}

const PARAM_NAME_OPTIONS = [
  "learning_rate", "batch_size", "hidden_dim", "num_layers",
  "attention_heads", "dropout", "weight_decay", "mlp_hidden_dim",
];

/**
 * Convert a CLI flag like "--sft-data-path" to a Python kwarg name "sft_data_path".
 * Strips leading dashes and replaces remaining dashes with underscores.
 */
function flagToArgName(flag: string): string {
  return flag.replace(/^--/, "").replace(/-/g, "_");
}

/** Return a human-readable label from a CLI flag. */
function flagToLabel(flag: string): string {
  return flag
    .replace(/^--/, "")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Check if a flag represents a file/folder path field. */
function isPathField(flag: string): boolean {
  return flag.includes("-path") || flag.includes("-dir");
}

function parseBestModelPath(output: string): string | null {
  try {
    const lines = output.split("\n");
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith("{") && line.includes("best_trial_id")) {
        const data = JSON.parse(line);
        const bestTrial = data.trials?.find(
          (t: { trial_id: number }) => t.trial_id === data.best_trial_id,
        );
        return bestTrial?.model_path || null;
      }
    }
  } catch { /* ignore parse errors */ }
  return null;
}

export function SweepConfigForm() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const registerCommand = useForgeCommand();
  const [method, setMethod] = useState<TrainingMethod>("train");
  const [dataset, setDataset] = useState("");
  const [outputDir, setOutputDir] = useState("./outputs/sweep");
  const [strategy, setStrategy] = useState("grid");
  const [maxTrials, setMaxTrials] = useState("10");
  const [metric, setMetric] = useState("validation_loss");
  const [maximize, setMaximize] = useState(false);
  const [params, setParams] = useState<SweepParam[]>([
    { name: "learning_rate", values: "0.001, 0.01, 0.1" },
  ]);
  const [methodArgs, setMethodArgs] = useState<Record<string, string>>({});
  const [results, setResults] = useState<string>("");
  const [registerModel, setRegisterModel] = useState(false);
  const [modelName, setModelName] = useState("My-Model-0");
  const [registered, setRegistered] = useState(false);

  /** Get the required fields for the current method, excluding --dataset (handled separately). */
  const requiredMethodFields = useMemo(
    () => REQUIRED_METHOD_FIELDS[method].filter((f) => f !== "--dataset"),
    [method],
  );

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

  function updateMethodArg(flag: string, value: string) {
    setMethodArgs((prev) => ({ ...prev, [flag]: value }));
  }

  async function startSweep() {
    if (!dataRoot) return;
    const paramDefs = params.map((p) => ({
      name: p.name,
      values: p.values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v)),
    }));
    const paramsJson = JSON.stringify({ parameters: paramDefs });
    const args = [
      "sweep", "--dataset", dataset, "--output-dir", outputDir,
      "--params", paramsJson, "--strategy", strategy,
      "--max-trials", maxTrials, "--metric", metric,
      "--method", method, "--json",
    ];
    if (maximize) args.push("--maximize");
    // Build method-args JSON from filled required fields
    if (requiredMethodFields.length > 0) {
      const mArgs: Record<string, string> = {};
      for (const flag of requiredMethodFields) {
        const argName = flagToArgName(flag);
        const value = methodArgs[flag];
        if (value?.trim()) {
          mArgs[argName] = value.trim();
        }
      }
      if (Object.keys(mArgs).length > 0) {
        args.push("--method-args", JSON.stringify(mArgs));
      }
    }
    setRegistered(false);
    const status = await command.run(dataRoot, args);
    if (status.status === "completed" && status.stdout) {
      setResults(status.stdout);
      if (registerModel && modelName.trim() && dataRoot) {
        const bestPath = parseBestModelPath(status.stdout);
        if (bestPath) {
          const regStatus = await registerCommand.run(dataRoot, [
            "model", "register", "--model-path", bestPath, "--tag", modelName.trim(),
          ]);
          if (regStatus.status === "completed" && regStatus.exit_code === 0) {
            setRegistered(true);
          }
        }
      }
    }
  }

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!dataset.trim()) m.push("dataset");
    if (!outputDir.trim()) m.push("output directory");
    if (params.length === 0) m.push("parameters");
    const hasEmptyValues = params.some(
      (p) => p.values.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v)).length === 0,
    );
    if (params.length > 0 && hasEmptyValues) m.push("parameter values");
    if (!metric.trim()) m.push("metric");
    // Check required method-specific fields
    for (const flag of requiredMethodFields) {
      if (!(methodArgs[flag] ?? "").trim()) {
        m.push(flagToLabel(flag).toLowerCase());
      }
    }
    return m;
  }, [dataset, outputDir, params, metric, requiredMethodFields, methodArgs]);

  if (results) {
    return (
      <SweepResultsView
        output={results}
        onBack={() => setResults("")}
        registeredAs={registered ? modelName : null}
      />
    );
  }

  return (
    <CommandFormPanel
      title="Hyperparameter Sweep"
      missing={missing}
      isRunning={command.isRunning}
      submitLabel="Start Sweep"
      runningLabel="Running Sweep..."
      onSubmit={() => startSweep().catch(console.error)}
      error={command.error}
    >
      <FormField label="Training Method">
        <select
          value={method}
          onChange={(e) => {
            setMethod(e.currentTarget.value as TrainingMethod);
            setMethodArgs({});
          }}
        >
          {TRAINING_METHODS.map((m) => (
            <option key={m.id} value={m.id}>{m.name}</option>
          ))}
        </select>
      </FormField>

      {requiredMethodFields.length > 0 && (
        <div className="grid-2">
          {requiredMethodFields.map((flag) => (
            <FormField key={flag} label={flagToLabel(flag)} required>
              {isPathField(flag) ? (
                <PathInput
                  value={methodArgs[flag] ?? ""}
                  onChange={(v) => updateMethodArg(flag, v)}
                  kind="file"
                />
              ) : (
                <input
                  value={methodArgs[flag] ?? ""}
                  onChange={(e) => updateMethodArg(flag, e.currentTarget.value)}
                />
              )}
            </FormField>
          ))}
        </div>
      )}

      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={dataset} onChange={setDataset} />
        </FormField>
        <FormField label="Output Directory" required>
          <PathInput value={outputDir} onChange={setOutputDir} kind="folder" />
        </FormField>
      </div>

      <div className="stack-sm">
        <div className="row-between">
          <span className="text-sm text-secondary">Parameters *</span>
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
              <FormField label="Values (comma-separated)" required>
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
        <FormField label="Metric" required>
          <MetricSelect value={metric} onChange={setMetric} />
        </FormField>
        <FormField label="Direction">
          <select value={maximize ? "maximize" : "minimize"} onChange={(e) => setMaximize(e.currentTarget.value === "maximize")}>
            <option value="minimize">Minimize</option>
            <option value="maximize">Maximize</option>
          </select>
        </FormField>
      </div>

      <FormField label="Register best model">
        <input
          type="checkbox"
          checked={registerModel}
          onChange={(e) => setRegisterModel(e.target.checked)}
          style={{ width: "auto" }}
        />
      </FormField>
      {registerModel && (
        <FormField label="Model Name" required>
          <input
            value={modelName}
            onChange={(e) => setModelName(e.currentTarget.value)}
            placeholder="My-Model-0"
          />
        </FormField>
      )}
    </CommandFormPanel>
  );
}
