import { FormField } from "../../../components/shared/FormField";

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
        <FormField label="Base Model Path">
          <input value={extra["--base-model-path"] ?? ""} onChange={(e) => update("--base-model-path", e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
        </FormField>
        <FormField label="Reference Data Path">
          <input value={extra["--reference-data-path"] ?? ""} onChange={(e) => update("--reference-data-path", e.currentTarget.value)} placeholder="optional — for drift detection" />
        </FormField>
        <FormField label="Drift Check Interval (epochs)">
          <input type="number" value={extra["--drift-check-interval"] ?? "1"} onChange={(e) => update("--drift-check-interval", e.currentTarget.value)} />
        </FormField>
        <FormField label="Max Perplexity Increase">
          <input value={extra["--max-perplexity-increase"] ?? "1.5"} onChange={(e) => update("--max-perplexity-increase", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
