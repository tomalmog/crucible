import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface BasicTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function BasicTrainForm({ extra, setExtra }: BasicTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Basic Training</h4>
      <p className="text-tertiary page-description">
        Standard supervised training from dataset. Configure shared parameters below.
      </p>
      <div className="grid-2">
        <FormField label="Dataset" required>
          <input value={extra["--dataset"] ?? ""} onChange={(e) => update("--dataset", e.currentTarget.value)} placeholder="dataset name" />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to resume from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path">
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder="auto-detect" filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
