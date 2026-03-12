import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface DistillTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function DistillTrainForm({ extra, setExtra }: DistillTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Teacher Model" required>
          <ModelSelect value={extra["--teacher-model-path"] ?? ""} onChange={(v) => update("--teacher-model-path", v)} />
        </FormField>
        <FormField label="Student Model">
          <ModelSelect value={extra["--student-model-path"] ?? ""} onChange={(v) => update("--student-model-path", v)} placeholder="optional — trains new student if empty" />
        </FormField>
        <FormField label="Temperature">
          <input value={extra["--temperature"] ?? "2.0"} onChange={(e) => update("--temperature", e.currentTarget.value)} />
        </FormField>
        <FormField label="Alpha">
          <input value={extra["--alpha"] ?? "0.5"} onChange={(e) => update("--alpha", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path">
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder="auto-detect" filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
