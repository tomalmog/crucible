import { DatasetSelect } from "../../components/shared/DatasetSelect";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { ModelSelect } from "../../components/shared/ModelSelect";
import type { RemoteInferenceConfig } from "../../hooks/useRemoteChatConfig";
import type { ClusterConfig } from "../../types/remote";
import type { SamplingPreset } from "./chatPersistence";

interface RemoteChatPanelState {
  clusterInfo: ClusterConfig | null;
  isSlurm: boolean;
  config: RemoteInferenceConfig;
  setPartition: (value: string) => void;
  setGpuType: (value: string) => void;
  setMemory: (value: string) => void;
  setTimeLimit: (value: string) => void;
}

interface ChatConfigPanelProps {
  modelPath: string;
  onModelPathChange: (value: string) => void;
  isRemoteModel: boolean;
  datasetName: string;
  onDatasetNameChange: (value: string) => void;
  tokenizerPath: string;
  onTokenizerPathChange: (value: string) => void;
  weightsPath: string;
  onWeightsPathChange: (value: string) => void;
  remote: RemoteChatPanelState;
  samplingPreset: SamplingPreset;
  onSamplingPresetChange: (value: SamplingPreset) => void;
  maxNewTokens: string;
  temperature: string;
  topK: string;
  onSamplingFieldChange: (field: string, value: string) => void;
  maxTokenLength: string;
  onMaxTokenLengthChange: (value: string) => void;
  positionEmbeddingType: string;
  onPositionEmbeddingTypeChange: (value: string) => void;
}

export function ChatConfigPanel(props: ChatConfigPanelProps) {
  return (
    <FormSection title="Model Configuration" defaultOpen>
      <div className="chat-config-grid">
        <FormField label="Model">
          <ModelSelect value={props.modelPath} onChange={props.onModelPathChange} />
        </FormField>
        <LocalModelFields {...props} />
        <RemoteModelFields remote={props.remote} isRemoteModel={props.isRemoteModel} />
        <SamplingFields {...props} />
        <LocalAdvancedFields {...props} />
      </div>
    </FormSection>
  );
}

function LocalModelFields(props: ChatConfigPanelProps) {
  if (props.isRemoteModel) return null;
  return (
    <>
      <FormField label="Dataset (optional)">
        <DatasetSelect
          value={props.datasetName}
          onChange={props.onDatasetNameChange}
          placeholder="optional"
        />
      </FormField>
      <FormField label="Tokenizer Path">
        <input
          value={props.tokenizerPath}
          onChange={(event) => props.onTokenizerPathChange(event.currentTarget.value)}
          placeholder="auto-detect"
        />
      </FormField>
      <FormField label="Custom Weights">
        <input
          value={props.weightsPath}
          onChange={(event) => props.onWeightsPathChange(event.currentTarget.value)}
          placeholder="optional .pt or .safetensors path"
        />
      </FormField>
    </>
  );
}

function RemoteModelFields(props: { remote: RemoteChatPanelState; isRemoteModel: boolean }) {
  if (!props.isRemoteModel || !props.remote.clusterInfo || !props.remote.isSlurm) return null;
  return (
    <>
      <FormField label="Partition">
        <select
          value={props.remote.config.partition}
          onChange={(event) => props.remote.setPartition(event.currentTarget.value)}
        >
          <option value="">Default</option>
          {props.remote.clusterInfo.partitions.map((partition) => (
            <option key={partition} value={partition}>{partition}</option>
          ))}
        </select>
      </FormField>
      <FormField label="GPU Type">
        <select
          value={props.remote.config.gpuType}
          onChange={(event) => props.remote.setGpuType(event.currentTarget.value)}
        >
          <option value="">Any</option>
          {props.remote.clusterInfo.gpuTypes.map((gpuType) => (
            <option key={gpuType} value={gpuType}>{gpuType}</option>
          ))}
        </select>
      </FormField>
      <FormField label="Memory">
        <input
          value={props.remote.config.memory}
          onChange={(event) => props.remote.setMemory(event.currentTarget.value)}
        />
      </FormField>
      <FormField label="Time Limit">
        <input
          value={props.remote.config.timeLimit}
          onChange={(event) => props.remote.setTimeLimit(event.currentTarget.value)}
          placeholder="HH:MM:SS"
        />
      </FormField>
    </>
  );
}

function SamplingFields(props: ChatConfigPanelProps) {
  return (
    <>
      <FormField label="Sampling Preset">
        <select
          value={props.samplingPreset}
          onChange={(event) => props.onSamplingPresetChange(event.currentTarget.value as SamplingPreset)}
        >
          <option value="deterministic">deterministic</option>
          <option value="balanced">balanced</option>
          <option value="creative">creative</option>
          <option value="custom">custom</option>
        </select>
      </FormField>
      <FormField label="Max New Tokens">
        <input
          value={props.maxNewTokens}
          onChange={(event) => props.onSamplingFieldChange("maxNewTokens", event.currentTarget.value)}
        />
      </FormField>
      <FormField label="Temperature">
        <input
          value={props.temperature}
          onChange={(event) => props.onSamplingFieldChange("temperature", event.currentTarget.value)}
        />
      </FormField>
      <FormField label="Top K">
        <input
          value={props.topK}
          onChange={(event) => props.onSamplingFieldChange("topK", event.currentTarget.value)}
        />
      </FormField>
    </>
  );
}

function LocalAdvancedFields(props: ChatConfigPanelProps) {
  if (props.isRemoteModel) return null;
  return (
    <>
      <FormField label="Max Token Length">
        <input
          value={props.maxTokenLength}
          onChange={(event) => props.onMaxTokenLengthChange(event.currentTarget.value)}
        />
      </FormField>
      <FormField label="Position Embedding">
        <select
          value={props.positionEmbeddingType}
          onChange={(event) => props.onPositionEmbeddingTypeChange(event.currentTarget.value)}
        >
          <option value="learned">learned</option>
          <option value="sinusoidal">sinusoidal</option>
        </select>
      </FormField>
    </>
  );
}
