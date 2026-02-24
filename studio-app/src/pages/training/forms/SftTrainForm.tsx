import { FormField } from "../../../components/shared/FormField";

interface SftTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function SftTrainForm({ extra, setExtra }: SftTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Supervised Fine-Tuning</h4>
      <div className="grid-2">
        <FormField label="Data Path">
          <input value={extra["--data-path"] ?? ""} onChange={(e) => update("--data-path", e.currentTarget.value)} placeholder="/path/to/sft_data.jsonl" />
        </FormField>
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
