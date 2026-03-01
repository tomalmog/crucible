import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

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
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <PathInput value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to start from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
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
          <PathInput value={extra["--reference-model-path"] ?? ""} onChange={(v) => update("--reference-model-path", v)} placeholder="optional — defaults to policy model" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
      </div>
    </div>
  );
}
