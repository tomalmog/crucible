import { useState, useMemo } from "react";
import { useNavigate } from "react-router";
import { useCrucible } from "../../context/CrucibleContext";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { ModelSelect } from "../../components/shared/ModelSelect";
import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { startCrucibleCommand } from "../../api/studioApi";
import { buildRemoteInterpArgs } from "../../api/commandArgs";
import { useInterpLocation } from "../../hooks/useInterpLocation";
import { jobLabel } from "../../utils/jobLabels";

type SaeMode = "train" | "analyze";

interface SaeFormProps {
  prefill?: Record<string, unknown>;
}

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

  const { isRemote, clusterName } = useInterpLocation(modelPath);

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
      learningRate, sparsityCoeff, saePath, inputText, topK,
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
        const args = buildRemoteInterpArgs(clusterName, method, JSON.stringify(methodArgs));
        await startCrucibleCommand(dataRoot, args, lbl, cfg);
      } else {
        const args: string[] = [method, "--model-path", modelPath, "--output-dir", "./outputs/interp"];
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
      navigate("/jobs");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <CommandFormPanel
      title="Sparse Autoencoder"
      missing={missing}
      isRunning={submitting}
      submitLabel={isRemote ? "Submit to Cluster" : mode === "train" ? "Train SAE" : "Analyze"}
      runningLabel="Submitting..."
      onSubmit={() => submit().catch(console.error)}
      error={error}
    >
      <div className="filter-pills" style={{ marginBottom: 12 }}>
        <button className={`filter-pill${mode === "train" ? " active" : ""}`} onClick={() => setMode("train")}>Train</button>
        <button className={`filter-pill${mode === "analyze" ? " active" : ""}`} onClick={() => setMode("analyze")}>Analyze</button>
      </div>

      <div className="grid-2">
        <FormField label="Model" required>
          <ModelSelect value={modelPath} onChange={setModelPath} />
        </FormField>

        {mode === "train" && (
          <>
            <FormField label="Dataset" required>
              <DatasetSelect value={dataset} onChange={setDataset} />
            </FormField>
            <FormField label="Layer Index" hint="-1 = last layer">
              <input type="number" value={layerIndex} onChange={(e) => setLayerIndex(e.currentTarget.value)} />
            </FormField>
            <FormField label="Latent Dim" hint="0 = auto (4x hidden)">
              <input type="number" min={0} value={latentDim} onChange={(e) => setLatentDim(e.currentTarget.value)} />
            </FormField>
            <FormField label="Max Samples">
              <input type="number" min={1} value={maxSamples} onChange={(e) => setMaxSamples(e.currentTarget.value)} />
            </FormField>
            <FormField label="Epochs">
              <input type="number" min={1} value={epochs} onChange={(e) => setEpochs(e.currentTarget.value)} />
            </FormField>
            <FormField label="Learning Rate">
              <input type="number" step="0.0001" value={learningRate} onChange={(e) => setLearningRate(e.currentTarget.value)} />
            </FormField>
            <FormField label="Sparsity Coefficient">
              <input type="number" step="0.0001" value={sparsityCoeff} onChange={(e) => setSparsityCoeff(e.currentTarget.value)} />
            </FormField>
          </>
        )}

        {mode === "analyze" && (
          <>
            <FormField label="SAE Path" required hint="Path to trained .pt file">
              <input value={saePath} onChange={(e) => setSaePath(e.currentTarget.value)} placeholder="./outputs/interp/sae_model.pt" />
            </FormField>
            <FormField label="Dataset" hint="Select the training dataset to see what each feature responds to">
              <DatasetSelect value={dataset} onChange={setDataset} />
            </FormField>
            <FormField label="Top K Features">
              <input type="number" min={1} value={topK} onChange={(e) => setTopK(e.currentTarget.value)} />
            </FormField>
          </>
        )}
      </div>

      {mode === "analyze" && (
        <FormField label="Input Text" required>
          <textarea value={inputText} onChange={(e) => setInputText(e.currentTarget.value)} placeholder="Enter text to analyze..." rows={3} />
        </FormField>
      )}

      {isRemote && (
        <div className="info-banner">
          Remote model selected — job will run on cluster <strong>{clusterName}</strong>
        </div>
      )}
    </CommandFormPanel>
  );
}
