import { useState, useMemo, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { listClusters } from "../../api/remoteApi";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteEvalArgs } from "../../api/commandArgs";
import { evalLabel } from "../../utils/jobLabels";
import type { ClusterConfig } from "../../types/remote";

const ALL_BENCHMARKS = [
  "mmlu", "gsm8k", "hellaswag", "arc", "truthfulqa", "winogrande", "humaneval",
] as const;

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
      : new Set(ALL_BENCHMARKS),
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "",
  );
  const [cluster, setCluster] = useState(
    typeof prefill?.cluster === "string" ? prefill.cluster : "",
  );
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
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
  const [comboOpen, setComboOpen] = useState(false);
  const comboRef = useRef<HTMLDivElement>(null);

  // Fetch available clusters; set default only if no cluster is already selected
  useEffect(() => {
    if (!dataRoot) return;
    listClusters(dataRoot).then((c) => {
      setClusters(c);
      if (c.length > 0 && !cluster) setCluster(c[0].name);
    }).catch(() => setClusters([]));
  }, [dataRoot]); // eslint-disable-line react-hooks/exhaustive-deps

  // Close menu when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (comboRef.current && !comboRef.current.contains(e.target as Node)) {
        setComboOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

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

  function snapshotConfig(): Record<string, unknown> {
    return {
      page: "benchmarks",
      modelPath,
      baseModelPath,
      selectedBenchmarks: Array.from(selectedBenchmarks),
      maxSamples,
      cluster,
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
      const args = buildRemoteEvalArgs(cluster, modelPath, benchmarks, {
        modelName: selectedEntry?.modelName || undefined,
        baseModel: baseModelPath.trim() || undefined,
        maxSamples: maxSamples.trim() || undefined,
        partition: partition || undefined,
        gpusPerNode,
        gpuType: gpuType || undefined,
        memory,
        timeLimit,
      });
      const selectedModelName = selectedEntry?.modelName || "model";
      await startCrucibleCommand(dataRoot, args, evalLabel(selectedModelName), cfg);
      navigate("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  const selected = ALL_BENCHMARKS.filter((b) => selectedBenchmarks.has(b));

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
        <div className="bench-combo-wrap" ref={comboRef}>
          <div className="bench-combo-selected">
            {selected.map((b) => (
              <span key={b} className="bench-combo-tag" onClick={() => toggleBenchmark(b)}>
                {b} &times;
              </span>
            ))}
            <span
              className="bench-combo-tag"
              style={{ background: "transparent", border: "1px dashed var(--border)", color: "var(--text-tertiary)" }}
              onClick={() => {
                if (selected.length === ALL_BENCHMARKS.length) setSelectedBenchmarks(new Set());
                else setSelectedBenchmarks(new Set(ALL_BENCHMARKS));
              }}
            >
              {selected.length === ALL_BENCHMARKS.length ? "Clear All" : "Select All"}
            </span>
            <span
              className="bench-combo-tag"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
              onClick={() => setComboOpen((p) => !p)}
            >
              {comboOpen ? "Close" : "Edit"}
            </span>
          </div>
          {comboOpen && (
            <div className="bench-combo-menu" style={{ position: "relative", top: 0 }}>
              {ALL_BENCHMARKS.map((b) => (
                <button
                  key={b}
                  type="button"
                  className={`bench-combo-option ${selectedBenchmarks.has(b) ? "bench-combo-option--active" : ""}`}
                  onClick={() => toggleBenchmark(b)}
                >
                  <span className="bench-combo-option-check">
                    {selectedBenchmarks.has(b) ? "\u2713" : ""}
                  </span>
                  {b}
                </button>
              ))}
            </div>
          )}
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
