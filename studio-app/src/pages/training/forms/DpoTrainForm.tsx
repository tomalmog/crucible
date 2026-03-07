import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface DpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DpoTrainForm({ extra, setExtra }: DpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasModel = (extra["--base-model"] ?? "").trim().length > 0;

  return (
    <div className="stack-sm">
      <h4>Direct Preference Optimization</h4>
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} />
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
        <FormField label="Reference Model">
          <ModelSelect value={extra["--reference-model-path"] ?? ""} onChange={(v) => update("--reference-model-path", v)} placeholder="optional — defaults to policy model" />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasModel ? "auto-loaded from model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasModel ? "auto-loaded from model" : "auto-detect"} disabled={hasModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
