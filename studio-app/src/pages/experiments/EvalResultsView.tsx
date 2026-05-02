import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelMultiSelect } from "../../components/shared/ModelMultiSelect";
import { BenchmarkMultiSelect } from "../../components/shared/BenchmarkMultiSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { evalLabel } from "../../utils/jobLabels";

interface EvalResultsViewProps {
  prefill?: Record<string, unknown>;
}

export function EvalResultsView({ prefill }: EvalResultsViewProps) {
  const { dataRoot, models } = useCrucible();
  const navigate = useNavigate();

  const [selectedModels, setSelectedModels] = useState<Set<string>>(
    Array.isArray(prefill?.selectedModels)
      ? new Set(prefill.selectedModels as string[])
      : new Set<string>(),
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

  // For remote detection — use first selected model
  const firstModel = useMemo(() => [...selectedModels][0] ?? "", [selectedModels]);
  const { isRemote, clusterName, clusterBackend } = useInterpLocation(firstModel);
  const isSlurm = clusterBackend === "slurm";

  const missing = useMemo(() => {
    const m: string[] = [];
    if (selectedModels.size === 0) m.push("models");
    if (selectedBenchmarks.size === 0) m.push("benchmarks");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [selectedModels, selectedBenchmarks, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "benchmarks",
      selectedModels: Array.from(selectedModels),
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
      const modelPaths = Array.from(selectedModels);
      const benchmarks = Array.from(selectedBenchmarks).join(",");

      const firstEntry = models.find(
        (m) => m.remotePath === modelPaths[0] || m.modelPath === modelPaths[0],
      );
      const firstName = firstEntry?.modelName ?? modelPaths[0];
      const labelSuffix = modelPaths.length > 1 ? ` +${modelPaths.length - 1}` : "";
      const label = evalLabel(firstName + labelSuffix);

      const cfg = snapshotConfig();
      let args: string[];

      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = {
          model_paths: modelPaths.join(","),
          benchmarks,
        };
        if (maxSamples.trim()) methodArgs.max_samples = parseInt(maxSamples, 10);
        args = buildDispatchSpec("eval", methodArgs, clusterBackend as "slurm" | "ssh", {
          label,
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
          "--model-paths", modelPaths.join(","),
          "--benchmarks", benchmarks,
        ];
        if (maxSamples.trim()) args.push("--max-samples", maxSamples);
      }

      await startCrucibleCommand(dataRoot, args, label, cfg);
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
      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}

      <FormField label="Models" required>
        <ModelMultiSelect
          selected={selectedModels}
          onChange={setSelectedModels}
        />
      </FormField>

      <FormField label="Benchmarks" required>
        <BenchmarkMultiSelect
          selected={selectedBenchmarks}
          onChange={setSelectedBenchmarks}
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
