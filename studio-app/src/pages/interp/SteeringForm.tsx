import { useEffect, useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import {
  CompactFormCard,
  CompactInfoBanner,
} from "../../components/shared/CompactForm";
import { datasetColumns, startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { jobLabel } from "../../utils/jobLabels";
import { SteeringFormFields, type ComputeSource, type SteerMode } from "./SteeringFormFields";

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
    prefill?.computeSource === "dataset" ? "dataset"
      : prefill?.computeSource === "two-datasets" ? "two-datasets"
      : "simple",
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

  // Compute — two datasets
  const [positiveDataset, setPositiveDataset] = useState(
    typeof prefill?.positiveDataset === "string" ? prefill.positiveDataset : "",
  );
  const [negativeDataset, setNegativeDataset] = useState(
    typeof prefill?.negativeDataset === "string" ? prefill.negativeDataset : "",
  );

  // Compute — dataset (single dataset + two columns)
  const [dataset, setDataset] = useState(
    typeof prefill?.dataset === "string" ? prefill.dataset : "",
  );
  const [columns, setColumns] = useState<string[]>([]);
  const [positiveColumn, setPositiveColumn] = useState(
    typeof prefill?.positiveColumn === "string" ? prefill.positiveColumn : "",
  );
  const [negativeColumn, setNegativeColumn] = useState(
    typeof prefill?.negativeColumn === "string" ? prefill.negativeColumn : "",
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
  const [baseModel, setBaseModel] = useState(
    typeof prefill?.baseModel === "string" ? prefill.baseModel : "",
  );

  // Fetch columns when dataset changes
  useEffect(() => {
    if (!dataRoot || !dataset) {
      setColumns([]);
      return;
    }
    datasetColumns(dataRoot, dataset)
      .then((cols) => {
        setColumns(cols);
        // Auto-clear column selections if they're no longer valid
        setPositiveColumn((prev) => (cols.includes(prev) ? prev : ""));
        setNegativeColumn((prev) => (cols.includes(prev) ? prev : ""));
      })
      .catch(() => setColumns([]));
  }, [dataRoot, dataset]);

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (mode === "compute") {
      if (computeSource === "simple") {
        if (!positiveText.trim()) m.push("positive text");
        if (!negativeText.trim()) m.push("negative text");
      } else if (computeSource === "dataset") {
        if (!dataset.trim()) m.push("dataset");
        if (!positiveColumn) m.push("positive column");
        if (!negativeColumn) m.push("negative column");
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
      dataset, positiveColumn, negativeColumn, positiveDataset, negativeDataset,
      vectorPath, inputText, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    const tab = mode === "compute" ? "steer-compute" : "steer-apply";
    return {
      page: "interpretability", tab, steerMode: mode, computeSource,
      modelPath, positiveText, negativeText, dataset, positiveColumn, negativeColumn, positiveDataset, negativeDataset,
      layerIndex, maxSamples, vectorPath, inputText, coefficient, maxNewTokens, baseModel,
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
        if (baseModel.trim()) methodArgs.base_model = baseModel;
        if (mode === "compute") {
          if (computeSource === "simple") {
            methodArgs.positive_text = positiveText;
            methodArgs.negative_text = negativeText;
          } else if (computeSource === "dataset") {
            methodArgs.dataset = dataset;
            methodArgs.positive_column = positiveColumn;
            methodArgs.negative_column = negativeColumn;
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
        const args = buildDispatchSpec(method, methodArgs, clusterBackend as "slurm", {
          label: lbl,
          clusterName,
          config: cfg,
        });
        await startCrucibleCommand(dataRoot, args, lbl, cfg);
      } else {
        const args: string[] = [method, "--model-path", modelPath, "--output-dir", "./outputs/interp"];
        if (baseModel.trim()) args.push("--base-model", baseModel);
        if (mode === "compute") {
          if (computeSource === "simple") {
            args.push("--positive-text", positiveText, "--negative-text", negativeText);
          } else if (computeSource === "dataset") {
            args.push("--dataset", dataset, "--positive-column", positiveColumn, "--negative-column", negativeColumn);
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
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CompactFormCard
      title="Activation Steering"
      description="Keep compute and apply workflows in one card, with source-specific controls that only appear when needed."
      missing={missing}
      actionLabel={isRemote ? "Submit to Cluster" : mode === "compute" ? "Compute vector" : "Apply steering"}
      runningLabel="Submitting..."
      isRunning={submitting}
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <SteeringFormFields
        baseModel={baseModel}
        coefficient={coefficient}
        columns={columns}
        computeSource={computeSource}
        dataset={dataset}
        inputText={inputText}
        layerIndex={layerIndex}
        maxNewTokens={maxNewTokens}
        maxSamples={maxSamples}
        mode={mode}
        modelPath={modelPath}
        negativeColumn={negativeColumn}
        negativeDataset={negativeDataset}
        negativeText={negativeText}
        positiveColumn={positiveColumn}
        positiveDataset={positiveDataset}
        positiveText={positiveText}
        setBaseModel={setBaseModel}
        setCoefficient={setCoefficient}
        setComputeSource={setComputeSource}
        setDataset={setDataset}
        setInputText={setInputText}
        setLayerIndex={setLayerIndex}
        setMaxNewTokens={setMaxNewTokens}
        setMaxSamples={setMaxSamples}
        setMode={setMode}
        setModelPath={setModelPath}
        setNegativeColumn={setNegativeColumn}
        setNegativeDataset={setNegativeDataset}
        setNegativeText={setNegativeText}
        setPositiveColumn={setPositiveColumn}
        setPositiveDataset={setPositiveDataset}
        setPositiveText={setPositiveText}
        setVectorPath={setVectorPath}
        vectorPath={vectorPath}
      />

      {isRemote && (
        <CompactInfoBanner>
          Remote model selected: job will run on cluster <strong>{clusterName}</strong>.
        </CompactInfoBanner>
      )}
    </CompactFormCard>
  );
}
