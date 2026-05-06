import { DatasetSelect } from "../../components/shared/DatasetSelect";
import {
  CompactField,
  CompactInfoBanner,
  CompactInlineField,
  CompactToggleGroup,
} from "../../components/shared/CompactForm";
import { ModelSelect } from "../../components/shared/ModelSelect";

export type SteerMode = "compute" | "apply";
export type ComputeSource = "simple" | "dataset" | "two-datasets";

interface SteeringFormFieldsProps {
  baseModel: string;
  coefficient: string;
  columns: string[];
  computeSource: ComputeSource;
  dataset: string;
  inputText: string;
  layerIndex: string;
  maxNewTokens: string;
  maxSamples: string;
  mode: SteerMode;
  modelPath: string;
  negativeColumn: string;
  negativeDataset: string;
  negativeText: string;
  positiveColumn: string;
  positiveDataset: string;
  positiveText: string;
  setBaseModel: (value: string) => void;
  setCoefficient: (value: string) => void;
  setComputeSource: (value: ComputeSource) => void;
  setDataset: (value: string) => void;
  setInputText: (value: string) => void;
  setLayerIndex: (value: string) => void;
  setMaxNewTokens: (value: string) => void;
  setMaxSamples: (value: string) => void;
  setMode: (value: SteerMode) => void;
  setModelPath: (value: string) => void;
  setNegativeColumn: (value: string) => void;
  setNegativeDataset: (value: string) => void;
  setNegativeText: (value: string) => void;
  setPositiveColumn: (value: string) => void;
  setPositiveDataset: (value: string) => void;
  setPositiveText: (value: string) => void;
  setVectorPath: (value: string) => void;
  vectorPath: string;
}

const MODE_OPTIONS: ReadonlyArray<{ label: string; value: SteerMode }> = [
  { label: "Compute", value: "compute" },
  { label: "Apply", value: "apply" },
];

const SOURCE_OPTIONS: ReadonlyArray<{ label: string; value: ComputeSource }> = [
  { label: "Two texts", value: "simple" },
  { label: "Paired columns", value: "dataset" },
  { label: "Two datasets", value: "two-datasets" },
];

export function SteeringFormFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <>
      <CompactField label="Mode">
        <CompactToggleGroup
          label="Steering mode"
          onChange={props.setMode}
          options={MODE_OPTIONS}
          value={props.mode}
        />
      </CompactField>
      {props.mode === "compute" ? (
        <SteeringComputeFields {...props} />
      ) : (
        <SteeringApplyFields {...props} />
      )}
    </>
  );
}

function SteeringComputeFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <>
      <div className="platform-form-grid platform-form-grid-3">
        <CompactInlineField label="Model" required>
          <ModelSelect value={props.modelPath} onChange={props.setModelPath} />
        </CompactInlineField>
        <CompactInlineField hint="for LoRA and QLoRA" label="Base model">
          <input value={props.baseModel} onChange={(e) => props.setBaseModel(e.currentTarget.value)} placeholder="optional" />
        </CompactInlineField>
        <CompactInlineField label="Source">
          <CompactToggleGroup
            label="Compute source"
            onChange={props.setComputeSource}
            options={SOURCE_OPTIONS}
            value={props.computeSource}
          />
        </CompactInlineField>
      </div>
      {props.computeSource === "simple" && <SteeringTextPairFields {...props} />}
      {props.computeSource === "dataset" && <SteeringDatasetFields {...props} />}
      {props.computeSource === "two-datasets" && <SteeringTwoDatasetFields {...props} />}
      <div className="platform-form-grid platform-form-grid-2">
        <CompactInlineField hint="-1 = last" label="Layer">
          <input type="number" value={props.layerIndex} onChange={(e) => props.setLayerIndex(e.currentTarget.value)} />
        </CompactInlineField>
        <CompactInlineField label="Samples">
          <input type="number" min={1} value={props.maxSamples} onChange={(e) => props.setMaxSamples(e.currentTarget.value)} />
        </CompactInlineField>
      </div>
    </>
  );
}

function SteeringApplyFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <>
      <div className="platform-form-grid platform-form-grid-3">
        <CompactInlineField label="Model" required>
          <ModelSelect value={props.modelPath} onChange={props.setModelPath} />
        </CompactInlineField>
        <CompactInlineField hint="steering_vector.pt" label="Vector path" required>
          <input value={props.vectorPath} onChange={(e) => props.setVectorPath(e.currentTarget.value)} placeholder="./outputs/interp/steering_vector.pt" />
        </CompactInlineField>
        <CompactInlineField hint="for LoRA and QLoRA" label="Base model">
          <input value={props.baseModel} onChange={(e) => props.setBaseModel(e.currentTarget.value)} placeholder="optional" />
        </CompactInlineField>
      </div>
      <CompactField label="Input text" required>
        <textarea value={props.inputText} onChange={(e) => props.setInputText(e.currentTarget.value)} placeholder="Once upon a time" rows={3} />
      </CompactField>
      <div className="platform-form-grid platform-form-grid-2">
        <CompactInlineField label="Coefficient">
          <input type="number" step="0.1" value={props.coefficient} onChange={(e) => props.setCoefficient(e.currentTarget.value)} />
        </CompactInlineField>
        <CompactInlineField label="Max new tokens">
          <input type="number" min={1} value={props.maxNewTokens} onChange={(e) => props.setMaxNewTokens(e.currentTarget.value)} />
        </CompactInlineField>
      </div>
    </>
  );
}

function SteeringTextPairFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <div className="platform-form-grid platform-form-grid-2">
      <CompactInlineField label="Positive text" required>
        <textarea value={props.positiveText} onChange={(e) => props.setPositiveText(e.currentTarget.value)} rows={3} />
      </CompactInlineField>
      <CompactInlineField label="Negative text" required>
        <textarea value={props.negativeText} onChange={(e) => props.setNegativeText(e.currentTarget.value)} rows={3} />
      </CompactInlineField>
    </div>
  );
}

function SteeringDatasetFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <>
      <div className="platform-form-grid platform-form-grid-3">
        <CompactInlineField label="Dataset" required>
          <DatasetSelect value={props.dataset} onChange={props.setDataset} />
        </CompactInlineField>
        <CompactInlineField label="Positive column" required>
          <SteeringColumnSelect columns={props.columns} value={props.positiveColumn} onChange={props.setPositiveColumn} />
        </CompactInlineField>
        <CompactInlineField label="Negative column" required>
          <SteeringColumnSelect columns={props.columns} value={props.negativeColumn} onChange={props.setNegativeColumn} />
        </CompactInlineField>
      </div>
      {props.dataset.trim() && props.columns.length === 0 && (
        <CompactInfoBanner>
          Column names will populate here when the selected dataset exposes metadata fields.
        </CompactInfoBanner>
      )}
    </>
  );
}

function SteeringColumnSelect({
  columns,
  onChange,
  value,
}: {
  columns: string[];
  onChange: (value: string) => void;
  value: string;
}): React.ReactNode {
  return (
    <select value={value} onChange={(e) => onChange(e.currentTarget.value)}>
      <option value="">Select column...</option>
      {columns.map((column) => (
        <option key={column} value={column}>{column}</option>
      ))}
    </select>
  );
}

function SteeringTwoDatasetFields(props: SteeringFormFieldsProps): React.ReactNode {
  return (
    <div className="platform-form-grid platform-form-grid-2">
      <CompactInlineField label="Positive dataset" required>
        <DatasetSelect value={props.positiveDataset} onChange={props.setPositiveDataset} />
      </CompactInlineField>
      <CompactInlineField label="Negative dataset" required>
        <DatasetSelect value={props.negativeDataset} onChange={props.setNegativeDataset} />
      </CompactInlineField>
    </div>
  );
}
