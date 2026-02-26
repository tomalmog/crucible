import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface DomainAdaptFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DomainAdaptForm({ extra, setExtra }: DomainAdaptFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Domain Adaptation</h4>
      <div className="grid-2">
        <FormField label="Dataset" required>
          <input value={extra["--dataset"] ?? ""} onChange={(e) => update("--dataset", e.currentTarget.value)} placeholder="dataset name" />
        </FormField>
        <FormField label="Base Model Path" required>
          <PathInput value={extra["--base-model-path"] ?? ""} onChange={(v) => update("--base-model-path", v)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Reference Data Path">
          <PathInput value={extra["--reference-data-path"] ?? ""} onChange={(v) => update("--reference-data-path", v)} placeholder="optional — for drift detection" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
        </FormField>
        <FormField label="Drift Check Interval (epochs)">
          <input type="number" value={extra["--drift-check-interval"] ?? "1"} onChange={(e) => update("--drift-check-interval", e.currentTarget.value)} />
        </FormField>
        <FormField label="Max Perplexity Increase">
          <input value={extra["--max-perplexity-increase"] ?? "1.5"} onChange={(e) => update("--max-perplexity-increase", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path">
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder="auto-detect" filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
