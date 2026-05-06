import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import {
  CompactField,
  CompactFormCard,
  CompactInfoBanner,
  CompactInlineField,
  CompactToggleGroup,
} from "../../components/shared/CompactForm";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildDispatchSpec } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { jobLabel } from "../../utils/jobLabels";

type SaeMode = "train" | "analyze";

interface SaeFormProps {
  prefill?: Record<string, unknown>;
}

const SAE_MODE_OPTIONS: ReadonlyArray<{ label: string; value: SaeMode }> = [
  { label: "Train", value: "train" },
  { label: "Analyze", value: "analyze" },
];

export function SaeForm({ prefill }: SaeFormProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const [mode, setMode] = useState<SaeMode>(
    prefill?.saeMode === "analyze" ? "analyze" : "train",
  );

  // Shared
  const [modelPath, setModelPath] = useState(
    typeof prefill?.modelPath === "string" ? prefill.modelPath : "",
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Train mode
  const [dataset, setDataset] = useState(
    typeof prefill?.dataset === "string" ? prefill.dataset : "",
  );
  const [layerIndex, setLayerIndex] = useState(
    typeof prefill?.layerIndex === "string" ? prefill.layerIndex : "-1",
  );
  const [latentDim, setLatentDim] = useState(
    typeof prefill?.latentDim === "string" ? prefill.latentDim : "0",
  );
  const [maxSamples, setMaxSamples] = useState(
    typeof prefill?.maxSamples === "string" ? prefill.maxSamples : "500",
  );
  const [epochs, setEpochs] = useState(
    typeof prefill?.epochs === "string" ? prefill.epochs : "10",
  );
  const [learningRate, setLearningRate] = useState(
    typeof prefill?.learningRate === "string" ? prefill.learningRate : "0.001",
  );
  const [sparsityCoeff, setSparsityCoeff] = useState(
    typeof prefill?.sparsityCoeff === "string" ? prefill.sparsityCoeff : "0.001",
  );

  // Analyze mode
  const [saePath, setSaePath] = useState(
    typeof prefill?.saePath === "string" ? prefill.saePath : "",
  );
  const [inputText, setInputText] = useState(
    typeof prefill?.inputText === "string" ? prefill.inputText : "",
  );
  const [topK, setTopK] = useState(
    typeof prefill?.topK === "string" ? prefill.topK : "10",
  );
  const [baseModel, setBaseModel] = useState(
    typeof prefill?.baseModel === "string" ? prefill.baseModel : "",
  );

  const { isRemote, clusterName, clusterBackend } = useInterpLocation(modelPath);

  const missing = useMemo(() => {
    const m: string[] = [];
    if (!modelPath.trim()) m.push("model");
    if (mode === "train" && !dataset.trim()) m.push("dataset");
    if (mode === "analyze" && !saePath.trim()) m.push("SAE path");
    if (mode === "analyze" && !inputText.trim()) m.push("input text");
    if (isRemote && !clusterName) m.push("cluster");
    return m;
  }, [modelPath, mode, dataset, saePath, inputText, isRemote, clusterName]);

  function snapshotConfig(): Record<string, unknown> {
    const tab = mode === "train" ? "sae-train" : "sae-analyze";
    return {
      page: "interpretability", tab, saeMode: mode,
      modelPath, dataset, layerIndex, latentDim, maxSamples, epochs,
      learningRate, sparsityCoeff, saePath, inputText, topK, baseModel,
    };
  }

  async function submit() {
    if (!dataRoot || missing.length > 0) return;
    setSubmitting(true);
    setError(null);
    try {
      const cfg = snapshotConfig();
      const method = mode === "train" ? "sae-train" : "sae-analyze";
      const lbl = jobLabel(method, modelPath);

      if (isRemote && clusterName) {
        const methodArgs: Record<string, unknown> = { model_path: modelPath, output_dir: "./outputs/interp" };
        if (baseModel.trim()) methodArgs.base_model = baseModel;
        if (mode === "train") {
          methodArgs.dataset_name = dataset;
          methodArgs.layer_index = parseInt(layerIndex || "-1", 10);
          methodArgs.latent_dim = parseInt(latentDim || "0", 10);
          methodArgs.max_samples = parseInt(maxSamples || "500", 10);
          methodArgs.epochs = parseInt(epochs || "10", 10);
          methodArgs.learning_rate = parseFloat(learningRate || "0.001");
          methodArgs.sparsity_coeff = parseFloat(sparsityCoeff || "0.001");
        } else {
          methodArgs.sae_path = saePath;
          methodArgs.input_text = inputText;
          methodArgs.top_k_features = parseInt(topK || "10", 10);
          if (dataset.trim()) methodArgs.dataset_name = dataset;
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
        if (mode === "train") {
          args.push("--dataset", dataset, "--layer-index", layerIndex || "-1",
            "--latent-dim", latentDim || "0", "--max-samples", maxSamples || "500",
            "--epochs", epochs || "10", "--learning-rate", learningRate || "0.001",
            "--sparsity-coeff", sparsityCoeff || "0.001");
        } else {
          args.push("--sae-path", saePath, "--input-text", inputText,
            "--top-k-features", topK || "10");
          if (dataset.trim()) args.push("--dataset", dataset);
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
      title="Sparse Autoencoder"
      missing={missing}
      description="Switch between training and feature inspection without losing the dense, compact layout."
      actionLabel={isRemote ? "Submit to Cluster" : mode === "train" ? "Train SAE" : "Analyze SAE"}
      runningLabel="Submitting..."
      isRunning={submitting}
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <CompactField label="Mode">
        <CompactToggleGroup
          label="SAE mode"
          onChange={setMode}
          options={SAE_MODE_OPTIONS}
          value={mode}
        />
      </CompactField>

      {mode === "train" ? (
        <>
          <div className="platform-form-grid platform-form-grid-3">
            <CompactInlineField label="Model" required>
              <ModelSelect value={modelPath} onChange={setModelPath} />
            </CompactInlineField>
            <CompactInlineField label="Dataset" required>
              <DatasetSelect value={dataset} onChange={setDataset} />
            </CompactInlineField>
            <CompactInlineField hint="for LoRA and QLoRA" label="Base model">
              <input value={baseModel} onChange={(e) => setBaseModel(e.currentTarget.value)} placeholder="optional" />
            </CompactInlineField>
          </div>
          <div className="platform-form-grid platform-form-grid-4">
            <CompactInlineField hint="-1 = last" label="Layer">
              <input type="number" value={layerIndex} onChange={(e) => setLayerIndex(e.currentTarget.value)} />
            </CompactInlineField>
            <CompactInlineField hint="0 = auto" label="Latent dim">
              <input type="number" min={0} value={latentDim} onChange={(e) => setLatentDim(e.currentTarget.value)} />
            </CompactInlineField>
            <CompactInlineField label="Samples">
              <input type="number" min={1} value={maxSamples} onChange={(e) => setMaxSamples(e.currentTarget.value)} />
            </CompactInlineField>
            <CompactInlineField label="Epochs">
              <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(e.currentTarget.value)} />
            </CompactInlineField>
          </div>
          <div className="platform-form-grid platform-form-grid-2">
            <CompactInlineField label="Learning rate">
              <input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(e.currentTarget.value)} />
            </CompactInlineField>
            <CompactInlineField label="Sparsity coeff.">
              <input type="number" step="0.0001" value={sparsityCoeff} onChange={(e) => setSparsityCoeff(e.currentTarget.value)} />
            </CompactInlineField>
          </div>
        </>
      ) : (
        <>
          <div className="platform-form-grid platform-form-grid-3">
            <CompactInlineField label="Model" required>
              <ModelSelect value={modelPath} onChange={setModelPath} />
            </CompactInlineField>
            <CompactInlineField hint="trained .pt file" label="SAE path" required>
              <input value={saePath} onChange={(e) => setSaePath(e.currentTarget.value)} placeholder="./outputs/interp/sae_model.pt" />
            </CompactInlineField>
            <CompactInlineField hint="for LoRA and QLoRA" label="Base model">
              <input value={baseModel} onChange={(e) => setBaseModel(e.currentTarget.value)} placeholder="optional" />
            </CompactInlineField>
          </div>
          <CompactField label="Input text" required>
          <textarea value={inputText} onChange={(e) => setInputText(e.currentTarget.value)} placeholder="Enter text to analyze..." rows={3} />
          </CompactField>
          <div className="platform-form-grid platform-form-grid-2">
            <CompactInlineField hint="optional context" label="Dataset">
              <DatasetSelect value={dataset} onChange={setDataset} />
            </CompactInlineField>
            <CompactInlineField label="Top K features">
              <input type="number" min={1} value={topK} onChange={(e) => setTopK(e.currentTarget.value)} />
            </CompactInlineField>
          </div>
        </>
      )}

      {isRemote && (
        <CompactInfoBanner>
          Remote model selected: job will run on cluster <strong>{clusterName}</strong>.
        </CompactInfoBanner>
      )}
    </CompactFormCard>
  );
}
