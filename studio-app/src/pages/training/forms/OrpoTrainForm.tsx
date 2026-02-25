import { FormField } from "../../../components/shared/FormField";

interface OrpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function OrpoTrainForm({ extra, setExtra }: OrpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Odds Ratio Preference Optimization</h4>
      <div className="grid-2">
        <FormField label="ORPO Data Path" required>
          <input value={extra["--orpo-data-path"] ?? ""} onChange={(e) => update("--orpo-data-path", e.currentTarget.value)} placeholder="/path/to/preferences.jsonl" />
        </FormField>
        <FormField label="Base Model" required>
          <input value={extra["--base-model"] ?? ""} onChange={(e) => update("--base-model", e.currentTarget.value)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <input value={extra["--initial-weights-path"] ?? ""} onChange={(e) => update("--initial-weights-path", e.currentTarget.value)} placeholder="optional — .pt checkpoint to start from" />
        </FormField>
        <FormField label="Lambda (odds-ratio weight)">
          <input value={extra["--lambda-orpo"] ?? "1.0"} onChange={(e) => update("--lambda-orpo", e.currentTarget.value)} />
        </FormField>
        <FormField label="Beta">
          <input value={extra["--beta"] ?? "0.1"} onChange={(e) => update("--beta", e.currentTarget.value)} />
        </FormField>
      </div>
    </div>
  );
}
