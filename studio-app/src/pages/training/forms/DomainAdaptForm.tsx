import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
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
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model-path"] ?? ""} onChange={(v) => update("--base-model-path", v)} />
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
