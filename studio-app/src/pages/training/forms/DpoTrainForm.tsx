import { FormField } from "../../../components/shared/FormField";

interface DpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DpoTrainForm({ extra, setExtra }: DpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Direct Preference Optimization</h4>
      <div className="grid-2">
        <FormField label="DPO Data Path">
          <input value={extra["--dpo-data-path"] ?? ""} onChange={(e) => update("--dpo-data-path", e.currentTarget.value)} placeholder="/path/to/preferences.jsonl" />
        </FormField>
        <FormField label="Base Model">
          <input value={extra["--base-model"] ?? ""} onChange={(e) => update("--base-model", e.currentTarget.value)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <input value={extra["--initial-weights-path"] ?? ""} onChange={(e) => update("--initial-weights-path", e.currentTarget.value)} placeholder="optional — .pt checkpoint to start from" />
        </FormField>
        <FormField label="Beta">
          <input value={extra["--beta"] ?? "0.1"} onChange={(e) => update("--beta", e.currentTarget.value)} />
        </FormField>
        <FormField label="Label Smoothing">
          <input value={extra["--label-smoothing"] ?? "0.0"} onChange={(e) => update("--label-smoothing", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reference Model Path">
          <input value={extra["--reference-model-path"] ?? ""} onChange={(e) => update("--reference-model-path", e.currentTarget.value)} placeholder="optional — defaults to policy model" />
        </FormField>
        <FormField label="Tokenizer Path">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
