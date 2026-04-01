import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface SftTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function SftTrainForm({ extra, setExtra }: SftTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasModel = (extra["--base-model"] ?? "").trim().length > 0;

  return (
    <div className="stack-sm">
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to resume from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasModel ? "auto-loaded from model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasModel ? "auto-loaded from model" : "auto-detect"} disabled={hasModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
        <FormField label="Mask Prompt Tokens" hint="mask loss on prompt tokens (instruction-tuning)">
          <label className="toggle-row">
            <input type="checkbox" checked={(extra["--mask-prompt-tokens"] ?? "true") === "true"} onChange={(e) => update("--mask-prompt-tokens", e.currentTarget.checked ? "true" : "false")} />
          </label>
        </FormField>
        <FormField label="Packing" hint="pack multiple examples into a single sequence">
          <label className="toggle-row">
            <input type="checkbox" checked={extra["--packing"] === "true"} onChange={(e) => update("--packing", e.currentTarget.checked ? "true" : "false")} />
          </label>
        </FormField>
      </div>
    </div>
  );
}
