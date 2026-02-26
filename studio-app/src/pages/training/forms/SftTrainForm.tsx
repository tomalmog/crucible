import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

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
          <PathInput value={extra["--sft-data-path"] ?? ""} onChange={(v) => update("--sft-data-path", v)} placeholder="/path/to/sft_data.jsonl" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
        </FormField>
        <FormField label="Base Model" required>
          <PathInput value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to resume from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasHfModel ? "auto-loaded from base model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasHfModel ? "auto-loaded from base model" : "auto-detect"} disabled={hasHfModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
