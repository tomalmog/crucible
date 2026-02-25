import { FormField } from "../../../components/shared/FormField";

interface KtoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function KtoTrainForm({ extra, setExtra }: KtoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Kahneman-Tversky Optimization</h4>
      <div className="grid-2">
        <FormField label="KTO Data Path">
          <input value={extra["--kto-data-path"] ?? ""} onChange={(e) => update("--kto-data-path", e.currentTarget.value)} placeholder="/path/to/kto_data.jsonl" />
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
        <FormField label="Desirable Weight">
          <input value={extra["--desirable-weight"] ?? "1.0"} onChange={(e) => update("--desirable-weight", e.currentTarget.value)} />
        </FormField>
        <FormField label="Undesirable Weight">
          <input value={extra["--undesirable-weight"] ?? "1.0"} onChange={(e) => update("--undesirable-weight", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reference Model Path">
          <input value={extra["--reference-model-path"] ?? ""} onChange={(e) => update("--reference-model-path", e.currentTarget.value)} placeholder="optional — defaults to policy model" />
        </FormField>
      </div>
    </div>
  );
}
