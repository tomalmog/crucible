import { FormField } from "../../../components/shared/FormField";

interface DpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DpoTrainForm({ extra, setExtra }: DpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Direct Preference Optimization</h4>
      <div className="grid-2">
        <FormField label="Preference Data Path">
          <input value={extra["--preference-data-path"] ?? ""} onChange={(e) => update("--preference-data-path", e.currentTarget.value)} placeholder="/path/to/preferences.jsonl" />
        </FormField>
        <FormField label="Beta">
          <input value={extra["--beta"] ?? "0.1"} onChange={(e) => update("--beta", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reference Model Path (optional)">
          <input value={extra["--ref-model-path"] ?? ""} onChange={(e) => update("--ref-model-path", e.currentTarget.value)} placeholder="same as model" />
        </FormField>
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
