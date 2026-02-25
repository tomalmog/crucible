import { FormField } from "../../../components/shared/FormField";

interface SftTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function SftTrainForm({ extra, setExtra }: SftTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasHfModel = (extra["--base-model"] ?? "").trim().length > 0;

  return (
    <div className="stack-sm">
      <h4>Supervised Fine-Tuning</h4>
      <div className="grid-2">
        <FormField label="SFT Data Path" required>
          <input value={extra["--sft-data-path"] ?? ""} onChange={(e) => update("--sft-data-path", e.currentTarget.value)} placeholder="/path/to/sft_data.jsonl" />
        </FormField>
        <FormField label="Base Model" required>
          <input value={extra["--base-model"] ?? ""} onChange={(e) => update("--base-model", e.currentTarget.value)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <input value={extra["--initial-weights-path"] ?? ""} onChange={(e) => update("--initial-weights-path", e.currentTarget.value)} placeholder="optional — .pt checkpoint to resume from" />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasHfModel ? "auto-loaded from base model" : undefined}>
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder={hasHfModel ? "auto-loaded from base model" : "auto-detect"} disabled={hasHfModel && !(extra["--tokenizer-path"] ?? "").trim()} />
        </FormField>
      </div>
    </div>
  );
}
