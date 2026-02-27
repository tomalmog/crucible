import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
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
      <h4>Knowledge Distillation</h4>
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Teacher Model Path" required>
          <PathInput value={extra["--teacher-model-path"] ?? ""} onChange={(v) => update("--teacher-model-path", v)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Student Model Path">
          <PathInput value={extra["--student-model-path"] ?? ""} onChange={(v) => update("--student-model-path", v)} placeholder="optional — HF model ID or path, trains new student if empty" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
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
