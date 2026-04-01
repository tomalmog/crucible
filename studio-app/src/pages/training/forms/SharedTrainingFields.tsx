import { SharedTrainingConfig } from "../../../types/training";
import { FormField } from "../../../components/shared/FormField";
import { FormSection } from "../../../components/shared/FormSection";
import { PathInput } from "../../../components/shared/PathInput";

interface SharedTrainingFieldsProps {
  config: SharedTrainingConfig;
  onChange: (config: SharedTrainingConfig) => void;
}

export function SharedTrainingFields({ config, onChange }: SharedTrainingFieldsProps) {
  function update(key: keyof SharedTrainingConfig, value: string) {
    onChange({ ...config, [key]: value });
  }

  return (
    <>
      <div className="grid-2">
        <FormField label="Epochs">
          <input type="number" value={config.epochs} onChange={(e) => update("epochs", e.currentTarget.value)} />
        </FormField>
        <FormField label="Learning Rate">
          <input value={config.learningRate} onChange={(e) => update("learningRate", e.currentTarget.value)} />
        </FormField>
        <FormField label="Batch Size">
          <input type="number" value={config.batchSize} onChange={(e) => update("batchSize", e.currentTarget.value)} />
        </FormField>
        <FormField label="Output Directory">
          <PathInput value={config.outputDir} onChange={(v) => update("outputDir", v)} kind="folder" />
        </FormField>
      </div>

      <FormSection title="Advanced Parameters">
        <div className="grid-3">
          <FormField label="Optimizer">
            <select value={config.optimizer} onChange={(e) => update("optimizer", e.currentTarget.value)}>
              <option value="adamw">AdamW</option>
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
            </select>
          </FormField>
          <FormField label="Precision">
            <select value={config.precision} onChange={(e) => update("precision", e.currentTarget.value)}>
              <option value="auto">Auto</option>
              <option value="fp32">FP32</option>
              <option value="fp16">FP16</option>
              <option value="bf16">BF16</option>
            </select>
          </FormField>
          <FormField label="Max Token Length">
            <input type="number" value={config.maxTokenLength} onChange={(e) => update("maxTokenLength", e.currentTarget.value)} />
          </FormField>
          <FormField label="Embedding Dim">
            <input type="number" value={config.embeddingDim} onChange={(e) => update("embeddingDim", e.currentTarget.value)} />
          </FormField>
          <FormField label="Num Heads">
            <input type="number" value={config.numHeads} onChange={(e) => update("numHeads", e.currentTarget.value)} />
          </FormField>
          <FormField label="Num Layers">
            <input type="number" value={config.numLayers} onChange={(e) => update("numLayers", e.currentTarget.value)} />
          </FormField>
          <FormField label="Checkpoint Every (0=off)">
            <input type="number" value={config.checkpointEvery} onChange={(e) => update("checkpointEvery", e.currentTarget.value)} />
          </FormField>
          <FormField label="MLP Hidden Dim">
            <input type="number" value={config.mlpHiddenDim} onChange={(e) => update("mlpHiddenDim", e.currentTarget.value)} />
          </FormField>
          <FormField label="MLP Layers">
            <input type="number" value={config.mlpLayers} onChange={(e) => update("mlpLayers", e.currentTarget.value)} />
          </FormField>
        </div>
      </FormSection>

      <FormSection title="Resume Training">
        <FormField label="Resume Checkpoint Path">
          <PathInput value={config.resumeCheckpointPath} onChange={(v) => update("resumeCheckpointPath", v)} placeholder="path/to/checkpoint" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
      </FormSection>

      <FormSection title="Experiment Tracking">
        <div className="grid-2">
          <FormField label="W&B Project (optional)">
            <input value={config.wandbProject} onChange={(e) => update("wandbProject", e.currentTarget.value)} placeholder="my-project" />
          </FormField>
          <FormField label="TensorBoard Dir (optional)">
            <PathInput value={config.tensorboardDir} onChange={(v) => update("tensorboardDir", v)} placeholder="./tb_logs" kind="folder" />
          </FormField>
        </div>
      </FormSection>
    </>
  );
}
