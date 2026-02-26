import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface DpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DpoTrainForm({ extra, setExtra }: DpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasHfModel = (extra["--base-model"] ?? "").trim().length > 0;

  return (
    <div className="stack-sm">
      <h4>Direct Preference Optimization</h4>
      <div className="grid-2">
        <FormField label="DPO Data Path" required>
          <PathInput value={extra["--dpo-data-path"] ?? ""} onChange={(v) => update("--dpo-data-path", v)} placeholder="/path/to/preferences.jsonl" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
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
        <FormField label="Label Smoothing">
          <input value={extra["--label-smoothing"] ?? "0.0"} onChange={(e) => update("--label-smoothing", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reference Model Path">
          <PathInput value={extra["--reference-model-path"] ?? ""} onChange={(v) => update("--reference-model-path", v)} placeholder="optional — defaults to policy model" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasHfModel ? "auto-loaded from base model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasHfModel ? "auto-loaded from base model" : "auto-detect"} disabled={hasHfModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
