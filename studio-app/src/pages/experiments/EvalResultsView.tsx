import { useState, useMemo, useCallback, useEffect } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { Search } from "lucide-react";
import { startCrucibleCommand, listBenchmarks } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { evalLabel } from "../../utils/jobLabels";

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
      : new Set<string>(),
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "",
  );
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
  const [registryBenchmarks, setRegistryBenchmarks] = useState<{ name: string; displayName: string }[]>([]);
  const [benchSearch, setBenchSearch] = useState("");

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);
  const isSlurm = clusterBackend === "slurm";

  // Load benchmarks from registry
  useEffect(() => {
    if (!dataRoot) return;
    listBenchmarks(dataRoot)
      .then((items) => setRegistryBenchmarks(items.map((b) => ({ name: b.name, displayName: b.displayName }))))
      .catch(() => setRegistryBenchmarks([]));
  }, [dataRoot]);

  const filteredBenchmarks = useMemo(() => {
    if (!benchSearch.trim()) return registryBenchmarks;
    const q = benchSearch.trim().toLowerCase();
    return registryBenchmarks.filter((b) => b.displayName.toLowerCase().includes(q) || b.name.includes(q));
  }, [registryBenchmarks, benchSearch]);

  const toggleBenchmark = useCallback((name: string) => {
    setSelectedBenchmarks((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (selectedBenchmarks.size === 0) m.push("benchmarks");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, selectedBenchmarks, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "benchmarks",
      modelPath,
      baseModelPath,
      selectedBenchmarks: Array.from(selectedBenchmarks),
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
      const benchmarks = Array.from(selectedBenchmarks).join(",");
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

  const allSelected = registryBenchmarks.length > 0 && registryBenchmarks.every((b) => selectedBenchmarks.has(b.name));

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

      <FormField label="Benchmarks" required>
        <div className="eval-checklist-toolbar">
          <div className="registry-search" style={{ maxWidth: 220 }}>
            <Search size={14} />
            <input
              value={benchSearch}
              onChange={(e) => setBenchSearch(e.currentTarget.value)}
              placeholder="Search benchmarks..."
            />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                if (allSelected) setSelectedBenchmarks(new Set());
                else setSelectedBenchmarks(new Set(registryBenchmarks.map((b) => b.name)));
              }}
            >
              {allSelected ? "Clear All" : "Select All"}
            </button>
            <span className="text-muted text-sm">{selectedBenchmarks.size} selected</span>
          </div>
        </div>
        <div className="eval-checklist">
          {filteredBenchmarks.map((b) => (
            <label key={b.name} className="eval-checklist-row">
              <input
                type="checkbox"
                checked={selectedBenchmarks.has(b.name)}
                onChange={() => toggleBenchmark(b.name)}
              />
              <span>{b.displayName}</span>
            </label>
          ))}
        </div>
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
