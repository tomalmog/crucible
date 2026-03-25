import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteInterpArgs, buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { jobLabel } from "../../utils/jobLabels";

type SteerMode = "compute" | "apply";
type ComputeSource = "simple" | "dataset";

interface SteeringFormProps {
  prefill?: Record<string, unknown>;
}

export function SteeringForm({ prefill }: SteeringFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [mode, setMode] = useState<SteerMode>(
    prefill?.steerMode === "apply" ? "apply" : "compute",
  );
  const [computeSource, setComputeSource] = useState<ComputeSource>(
    prefill?.computeSource === "dataset" ? "dataset" : "simple",
  );

  // Shared
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Compute — simple
  const [positiveText, setPositiveText] = useState(
    typeof prefill?.positiveText === "string" ? prefill.positiveText : "",
  );
  const [negativeText, setNegativeText] = useState(
    typeof prefill?.negativeText === "string" ? prefill.negativeText : "",
  );

  // Compute — dataset
  const [positiveDataset, setPositiveDataset] = useState(
    typeof prefill?.positiveDataset === "string" ? prefill.positiveDataset : "",
  );
  const [negativeDataset, setNegativeDataset] = useState(
    typeof prefill?.negativeDataset === "string" ? prefill.negativeDataset : "",
  );

  // Compute — shared
  const [layerIndex, setLayerIndex] = useState(
    typeof prefill?.layerIndex === "string" ? prefill.layerIndex : "-1",
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "100",
  );

  // Apply
  const [vectorPath, setVectorPath] = useState(
    typeof prefill?.vectorPath === "string" ? prefill.vectorPath : "",
  );
  const [inputText, setInputText] = useState(
    typeof prefill?.inputText === "string" ? prefill.inputText : "",
  );
  const [coefficient, setCoefficient] = useState(
    typeof prefill?.coefficient === "string" ? prefill.coefficient : "1.0",
  );
  const [maxNewTokens, setMaxNewTokens] = useState(
    typeof prefill?.maxNewTokens === "string" ? prefill.maxNewTokens : "50",
  );

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);
  const isSlurm = clusterBackend === "slurm";

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (mode === "compute") {
      if (computeSource === "simple") {
        if (!positiveText.trim()) m.push("positive text");
        if (!negativeText.trim()) m.push("negative text");
      } else {
        if (!positiveDataset.trim()) m.push("positive dataset");
        if (!negativeDataset.trim()) m.push("negative dataset");
      }
    }
    if (mode === "apply") {
      if (!vectorPath.trim()) m.push("vector path");
      if (!inputText.trim()) m.push("input text");
    }
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, mode, computeSource, positiveText, negativeText,
      positiveDataset, negativeDataset, vectorPath, inputText, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    const tab = mode === "compute" ? "steer-compute" : "steer-apply";
    return {
      page: "interpretability", tab, steerMode: mode, computeSource,
      modelPath, positiveText, negativeText, positiveDataset, negativeDataset,
      layerIndex, maxSamples, vectorPath, inputText, coefficient, maxNewTokens,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const cfg = snapshotConfig();
      const method = mode === "compute" ? "steer-compute" : "steer-apply";
      const lbl = jobLabel(method, modelPath);

      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = { model_path: modelPath, output_dir: "./outputs/interp" };
        if (mode === "compute") {
          if (computeSource === "simple") {
            methodArgs.positive_text = positiveText;
            methodArgs.negative_text = negativeText;
          } else {
            methodArgs.positive_dataset = positiveDataset;
            methodArgs.negative_dataset = negativeDataset;
          }
          methodArgs.layer_index = parseInt(layerIndex || "-1", 10);
          methodArgs.max_samples = parseInt(maxSamples || "100", 10);
        } else {
          methodArgs.steering_vector_path = vectorPath;
          methodArgs.input_text = inputText;
          methodArgs.coefficient = parseFloat(coefficient || "1.0");
          methodArgs.max_new_tokens = parseInt(maxNewTokens || "50", 10);
        }
        const args = isSlurm
          ? buildRemoteInterpArgs(clusterName, method, JSON.stringify(methodArgs))
          : buildDispatchSpec(method, methodArgs, clusterBackend as "ssh", {
              label: lbl,
              clusterName,
            });
        await startCrucibleCommand(dataRoot, args, lbl, cfg);
      } else {
        const args: string[] = [method, "--model-path", modelPath, "--output-dir", "./outputs/interp"];
        if (mode === "compute") {
          if (computeSource === "simple") {
            args.push("--positive-text", positiveText, "--negative-text", negativeText);
          } else {
            args.push("--positive-dataset", positiveDataset, "--negative-dataset", negativeDataset);
          }
          args.push("--layer-index", layerIndex || "-1", "--max-samples", maxSamples || "100");
        } else {
          args.push("--steering-vector-path", vectorPath, "--input-text", inputText,
            "--coefficient", coefficient || "1.0", "--max-new-tokens", maxNewTokens || "50");
        }
        await startCrucibleCommand(dataRoot, args, lbl, cfg);
      }
      navigate("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandFormPanel
      title="Activation Steering"
      missing={missing}
      isRunning={submitting}
      submitLabel={isRemote ? "Submit to Cluster" : mode === "compute" ? "Compute Vector" : "Apply Steering"}
      runningLabel="Submitting..."
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <div className="filter-pills" style={{ marginBottom: 12 }}>
        <button className={`filter-pill${mode === "compute" ? " active" : ""}`} onClick={() => setMode("compute")}>Compute</button>
        <button className={`filter-pill${mode === "apply" ? " active" : ""}`} onClick={() => setMode("apply")}>Apply</button>
      </div>

      <div className="grid-2">
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} />
        </FormField>

        {mode === "compute" && (
          <FormField label="Source">
            <select value={computeSource} onChange={(e) => setComputeSource(e.currentTarget.value as ComputeSource)}>
              <option value="simple">Simple (two texts)</option>
              <option value="dataset">Dataset (two datasets)</option>
            </select>
          </FormField>
        )}
      </div>

      {mode === "compute" && computeSource === "simple" && (
        <>
          <FormField label="Positive Text" required>
            <textarea value={positiveText} onChange={(e) => setPositiveText(e.currentTarget.value)}
              placeholder="Love, joy, happiness, kindness" rows={2} />
          </FormField>
          <FormField label="Negative Text" required>
            <textarea value={negativeText} onChange={(e) => setNegativeText(e.currentTarget.value)}
              placeholder="Hate, anger, sadness, cruelty" rows={2} />
          </FormField>
        </>
      )}

      {mode === "compute" && computeSource === "dataset" && (
        <div className="grid-2">
          <FormField label="Positive Dataset" required>
            <DatasetSelect value={positiveDataset} onChange={setPositiveDataset} />
          </FormField>
          <FormField label="Negative Dataset" required>
            <DatasetSelect value={negativeDataset} onChange={setNegativeDataset} />
          </FormField>
        </div>
      )}

      {mode === "compute" && (
        <div className="grid-2">
          <FormField label="Layer Index" hint="-1 = last layer">
            <input type="number" value={layerIndex} onChange={(e) => setLayerIndex(e.currentTarget.value)} />
          </FormField>
          <FormField label="Max Samples">
            <input type="number" min={1} value={maxSamples} onChange={(e) => setMaxSamples(e.currentTarget.value)} />
          </FormField>
        </div>
      )}

      {mode === "apply" && (
        <>
          <div className="grid-2">
            <FormField label="Vector Path" required hint="Path to steering_vector.pt">
              <input value={vectorPath} onChange={(e) => setVectorPath(e.currentTarget.value)}
                placeholder="./outputs/interp/steering_vector.pt" />
            </FormField>
            <FormField label="Coefficient">
              <input type="number" step="0.1" value={coefficient} onChange={(e) => setCoefficient(e.currentTarget.value)} />
            </FormField>
            <FormField label="Max New Tokens">
              <input type="number" min={1} value={maxNewTokens} onChange={(e) => setMaxNewTokens(e.currentTarget.value)} />
            </FormField>
          </div>
          <FormField label="Input Text" required>
            <textarea value={inputText} onChange={(e) => setInputText(e.currentTarget.value)}
              placeholder="Once upon a time" rows={3} />
          </FormField>
        </>
      )}

      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}
    </CommandFormPanel>
  );
}
