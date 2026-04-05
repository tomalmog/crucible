import { useState, useMemo, useCallback } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { evalLabel } from "../../utils/jobLabels";

// ── Benchmark catalog ────────────────────────────────────────────────

interface BenchmarkInfo {
  name: string;
  label: string;
  category: "knowledge" | "reasoning" | "code" | "commonsense";
}

const BENCHMARKS: BenchmarkInfo[] = [
  { name: "mmlu", label: "MMLU", category: "knowledge" },
  { name: "gpqa", label: "GPQA", category: "knowledge" },
  { name: "truthfulqa", label: "TruthfulQA", category: "knowledge" },
  { name: "gsm8k", label: "GSM8K", category: "reasoning" },
  { name: "math", label: "MATH", category: "reasoning" },
  { name: "bbh", label: "BBH", category: "reasoning" },
  { name: "arc", label: "ARC Challenge", category: "reasoning" },
  { name: "arc_easy", label: "ARC Easy", category: "reasoning" },
  { name: "hellaswag", label: "HellaSwag", category: "commonsense" },
  { name: "winogrande", label: "WinoGrande", category: "commonsense" },
  { name: "boolq", label: "BoolQ", category: "commonsense" },
  { name: "piqa", label: "PIQA", category: "commonsense" },
  { name: "openbookqa", label: "OpenBookQA", category: "commonsense" },
  { name: "humaneval", label: "HumanEval", category: "code" },
  { name: "mbpp", label: "MBPP", category: "code" },
];

interface Preset {
  label: string;
  description: string;
  benchmarks: string[];
}

const PRESETS: Preset[] = [
  { label: "Quick", description: "Fast sanity check", benchmarks: ["hellaswag", "arc_easy", "boolq", "piqa"] },
  { label: "Standard", description: "Open LLM Leaderboard set", benchmarks: ["mmlu", "hellaswag", "arc", "winogrande", "truthfulqa", "gsm8k"] },
  { label: "Reasoning", description: "Math & logic focused", benchmarks: ["mmlu", "gsm8k", "math", "bbh", "gpqa", "arc"] },
  { label: "Code", description: "Code generation", benchmarks: ["humaneval", "mbpp"] },
  { label: "Comprehensive", description: "All benchmarks", benchmarks: BENCHMARKS.map((b) => b.name) },
];

const CATEGORIES = ["knowledge", "reasoning", "commonsense", "code"] as const;
const CATEGORY_LABELS: Record<string, string> = {
  knowledge: "Knowledge",
  reasoning: "Reasoning",
  commonsense: "Commonsense",
  code: "Code",
};

// ── Component ────────────────────────────────────────────────────────

interface EvalResultsViewProps {
  prefill?: Record<string, unknown>;
}

export function EvalResultsView({ prefill }: EvalResultsViewProps) {
  const { dataRoot, models } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [baseModelPath, setBaseModelPath] = useState(
    typeof prefill?.baseModelPath === "string" ? prefill.baseModelPath : "",
  );
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<Set<string>>(
    Array.isArray(prefill?.selectedBenchmarks)
      ? new Set(prefill.selectedBenchmarks as string[])
      : new Set(PRESETS[0].benchmarks), // Default to Quick
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "",
  );
  const [customTasks, setCustomTasks] = useState("");
  const [partition, setPartition] = useState(
    typeof prefill?.partition === "string" ? prefill.partition : "",
  );
  const [gpusPerNode, setGpusPerNode] = useState(
    typeof prefill?.gpusPerNode === "string" ? prefill.gpusPerNode : "1",
  );
  const [gpuType, setGpuType] = useState(
    typeof prefill?.gpuType === "string" ? prefill.gpuType : "",
  );
  const [memory, setMemory] = useState(
    typeof prefill?.memory === "string" ? prefill.memory : "32G",
  );
  const [timeLimit, setTimeLimit] = useState(
    typeof prefill?.timeLimit === "string" ? prefill.timeLimit : "04:00:00",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);
  const isSlurm = clusterBackend === "slurm";

  const toggleBenchmark = useCallback((name: string) => {
    setSelectedBenchmarks((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const applyPreset = useCallback((preset: Preset) => {
    setSelectedBenchmarks(new Set(preset.benchmarks));
  }, []);

  // Combine selected catalog benchmarks + custom task names
  const allBenchmarks = useMemo(() => {
    const all = new Set(selectedBenchmarks);
    if (customTasks.trim()) {
      for (const t of customTasks.split(",").map((s) => s.trim()).filter(Boolean)) {
        all.add(t);
      }
    }
    return all;
  }, [selectedBenchmarks, customTasks]);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (allBenchmarks.size === 0) m.push("benchmarks");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, allBenchmarks, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "benchmarks",
      modelPath,
      baseModelPath,
      selectedBenchmarks: Array.from(allBenchmarks),
      maxSamples,
      cluster: clusterName,
      partition,
      gpusPerNode,
      gpuType,
      memory,
      timeLimit,
    };
  }

  async function submitEval() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const benchmarks = Array.from(allBenchmarks).join(",");
      const selectedEntry = models.find(
        (m) => m.remotePath === modelPath || m.modelPath === modelPath,
      );
      const cfg = snapshotConfig();
      const selectedModelName = selectedEntry?.modelName || "model";

      let args: string[];
      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = {
          model_path: modelPath,
          benchmarks,
        };
        if (baseModelPath.trim()) methodArgs.base_model_path = baseModelPath.trim();
        if (maxSamples.trim()) methodArgs.max_samples = parseInt(maxSamples, 10);
        args = buildDispatchSpec("eval", methodArgs, clusterBackend as "slurm" | "ssh", {
          label: evalLabel(selectedModelName),
          clusterName,
          resources: isSlurm ? {
            partition: partition || "",
            nodes: 1,
            gpus_per_node: parseInt(gpusPerNode, 10) || 1,
            gpu_type: gpuType || "",
            cpus_per_task: 4,
            memory,
            time_limit: timeLimit,
          } : undefined,
        });
      } else {
        args = [
          "eval",
          "--model-path", modelPath,
          "--benchmarks", benchmarks,
        ];
        if (baseModelPath.trim()) args.push("--base-model", baseModelPath.trim());
        if (maxSamples.trim()) args.push("--max-samples", maxSamples);
      }

      await startCrucibleCommand(dataRoot, args, evalLabel(selectedModelName), cfg);
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandFormPanel
      title="Model Evaluation"
      missing={missing}
      isRunning={submitting}
      submitLabel={isRemote ? "Submit to Cluster" : "Run Evaluation"}
      runningLabel="Submitting..."
      onSubmit={() => submitEval().catch(console.error)}
      error={error}
    >
      <div className="grid-2">
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} />
        </FormField>
        <FormField label="Base Model" hint="optional, for comparison">
          <ModelSelect
            value={baseModelPath}
            onChange={setBaseModelPath}
            placeholder="select base model (optional)"
          />
        </FormField>
      </div>

      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}

      {/* Benchmarks with presets inline */}
      <FormField label="">
        <div className="eval-benchmark-grid">
          <div className="eval-benchmark-category">
            <div className="eval-category-label">Presets</div>
            <div className="eval-category-items">
              {PRESETS.map((p) => {
                const isActive = p.benchmarks.length === selectedBenchmarks.size
                  && p.benchmarks.every((b) => selectedBenchmarks.has(b));
                return (
                  <span
                    key={p.label}
                    className={`bench-combo-tag ${isActive ? "bench-combo-tag--active" : ""}`}
                    onClick={() => applyPreset(p)}
                    title={p.description}
                  >
                    {p.label}
                  </span>
                );
              })}
            </div>
          </div>
          {CATEGORIES.map((cat) => (
            <div key={cat} className="eval-benchmark-category">
              <div className="eval-category-label">{CATEGORY_LABELS[cat]}</div>
              <div className="eval-category-items">
                {BENCHMARKS.filter((b) => b.category === cat).map((b) => (
                  <label key={b.name} className={`eval-benchmark-chip ${selectedBenchmarks.has(b.name) ? "eval-benchmark-chip--active" : ""}`}>
                    <input
                      type="checkbox"
                      checked={selectedBenchmarks.has(b.name)}
                      onChange={() => toggleBenchmark(b.name)}
                      style={{ display: "none" }}
                    />
                    {b.label}
                  </label>
                ))}
              </div>
            </div>
          ))}
        </div>
      </FormField>

      {/* Custom tasks */}
      <FormField label="Custom Tasks" hint="comma-separated lm-eval task names (14,000+ available)">
        <input
          value={customTasks}
          onChange={(e) => setCustomTasks(e.currentTarget.value)}
          placeholder="e.g. agieval, copa, social_iqa"
        />
      </FormField>

      <FormField label="Max Samples" hint="optional, leave empty for full dataset">
        <input
          type="number"
          value={maxSamples}
          onChange={(e) => setMaxSamples(e.currentTarget.value)}
          placeholder="e.g. 100"
        />
      </FormField>

      {isRemote && isSlurm && (
        <div className="grid-2">
          <FormField label="Partition">
            <select value={partition} onChange={(e) => setPartition(e.currentTarget.value)}>
              <option value="">Default</option>
            </select>
          </FormField>
          <FormField label="GPU Type">
            <select value={gpuType} onChange={(e) => setGpuType(e.currentTarget.value)}>
              <option value="">Any</option>
            </select>
          </FormField>
          <FormField label="GPUs">
            <input
              type="number"
              min={1}
              value={gpusPerNode}
              onChange={(e) => setGpusPerNode(e.currentTarget.value)}
            />
          </FormField>
          <FormField label="Memory">
            <input value={memory} onChange={(e) => setMemory(e.currentTarget.value)} />
          </FormField>
          <FormField label="Time Limit">
            <input
              value={timeLimit}
              onChange={(e) => setTimeLimit(e.currentTarget.value)}
              placeholder="HH:MM:SS"
            />
          </FormField>
        </div>
      )}
    </CommandFormPanel>
  );
}
