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
        <FormField label="Source Domain Data Path">
          <input value={extra["--source-data-path"] ?? ""} onChange={(e) => update("--source-data-path", e.currentTarget.value)} placeholder="/path/to/source_domain.jsonl" />
        </FormField>
        <FormField label="Target Domain Data Path">
          <input value={extra["--target-data-path"] ?? ""} onChange={(e) => update("--target-data-path", e.currentTarget.value)} placeholder="/path/to/target_domain.jsonl" />
        </FormField>
        <FormField label="Pre-trained Model Path">
          <input value={extra["--pretrained-model-path"] ?? ""} onChange={(e) => update("--pretrained-model-path", e.currentTarget.value)} placeholder="/path/to/pretrained.pt" />
        </FormField>
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
