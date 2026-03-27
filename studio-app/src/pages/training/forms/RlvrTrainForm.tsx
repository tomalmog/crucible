import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface RlvrTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function RlvrTrainForm({ extra, setExtra }: RlvrTrainFormProps) {
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
        <FormField label="Verifier Type">
          <select value={extra["--verifier-type"] ?? "code"} onChange={(e) => update("--verifier-type", e.currentTarget.value)}>
            <option value="code">Code</option>
            <option value="math">Math</option>
          </select>
        </FormField>
        <FormField label="Max Verification Attempts">
          <input type="number" value={extra["--max-attempts"] ?? "3"} onChange={(e) => update("--max-attempts", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reward (Correct)">
          <input value={extra["--reward-correct"] ?? "1.0"} onChange={(e) => update("--reward-correct", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reward (Incorrect)">
          <input value={extra["--reward-incorrect"] ?? "-0.5"} onChange={(e) => update("--reward-incorrect", e.currentTarget.value)} />
        </FormField>
      </div>
    </div>
  );
}
