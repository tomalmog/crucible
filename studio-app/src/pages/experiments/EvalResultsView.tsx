import { useState, useMemo, useEffect, useCallback } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { listClusters } from "../../api/remoteApi";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteEvalArgs } from "../../api/commandArgs";
import type { ClusterConfig } from "../../types/remote";

const ALL_BENCHMARKS = [
  "mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande", "humaneval",
] as const;

export function EvalResultsView() {
  const { dataRoot, modelGroups } = useCrucible();
  const navigate = useNavigate();
  const [modelPath, setModelPath] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("");
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<Set<string>>(
    new Set(ALL_BENCHMARKS),
  );
  const [maxSamples, setMaxSamples] = useState("");
  const [cluster, setCluster] = useState("");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [partition, setPartition] = useState("");
  const [gpusPerNode, setGpusPerNode] = useState("1");
  const [gpuType, setGpuType] = useState("");
  const [memory, setMemory] = useState("32G");
  const [timeLimit, setTimeLimit] = useState("04:00:00");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!dataRoot) return;
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0 && !cluster) setCluster(c[0].name);
    }).catch(() => setClusters([]));
  }, [dataRoot]);

  const selectedCluster = clusters.find((c) => c.name === cluster);

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
    if (!cluster) m.push("cluster");
    if (selectedBenchmarks.size === 0) m.push("benchmarks");
    return m;
  }, [modelPath, cluster, selectedBenchmarks]);

  async function submitEval() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const benchmarks = Array.from(selectedBenchmarks).join(",");
      // Look up the registered model name from the selected path
      const selectedGroup = modelGroups.find(
        (g) => g.activeRemotePath === modelPath || g.activeModelPath === modelPath,
      );
      const args = buildRemoteEvalArgs(cluster, modelPath, benchmarks, {
        modelName: selectedGroup?.modelName || undefined,
        baseModel: baseModelPath.trim() || undefined,
        maxSamples: maxSamples.trim() || undefined,
        partition: partition || undefined,
        gpusPerNode,
        gpuType: gpuType || undefined,
        memory,
        timeLimit,
      });
      await startCrucibleCommand(dataRoot, args);
      navigate("/jobs");
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
      submitLabel="Submit to Cluster"
      runningLabel="Submitting..."
      onSubmit={() => submitEval().catch(console.error)}
      error={error}
    >
      <div className="grid-2">
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} remoteOnly />
        </FormField>
        <FormField label="Base Model" hint="optional, for comparison">
          <ModelSelect
            value={baseModelPath}
            onChange={setBaseModelPath}
            placeholder="select base model (optional)"
            remoteOnly
          />
        </FormField>
      </div>

      <FormField label="Benchmarks" required>
        <div className="flex-row" style={{ flexWrap: "wrap", gap: 8 }}>
          {ALL_BENCHMARKS.map((b) => (
            <label
              key={b}
              style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 4, cursor: "pointer" }}
            >
              <input
                type="checkbox"
                checked={selectedBenchmarks.has(b)}
                onChange={() => toggleBenchmark(b)}
                style={{ width: "auto" }}
              />
              <span style={{ fontSize: "0.8125rem" }}>{b}</span>
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

      <div className="grid-2">
        <FormField label="Cluster" required>
          <select value={cluster} onChange={(e) => setCluster(e.currentTarget.value)}>
            {clusters.length === 0 && <option value="">No clusters registered</option>}
            {clusters.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.user}@{c.host})
              </option>
            ))}
          </select>
        </FormField>
        <FormField label="Partition">
          <select value={partition} onChange={(e) => setPartition(e.currentTarget.value)}>
            <option value="">Default</option>
            {selectedCluster?.partitions.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </FormField>
        <FormField label="GPU Type">
          <select value={gpuType} onChange={(e) => setGpuType(e.currentTarget.value)}>
            <option value="">Any</option>
            {selectedCluster?.gpuTypes.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
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
    </CommandFormPanel>
  );
}
