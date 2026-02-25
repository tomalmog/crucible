import { FormField } from "../../../components/shared/FormField";

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
        <FormField label="Teacher Model Path" required>
          <input value={extra["--teacher-model-path"] ?? ""} onChange={(e) => update("--teacher-model-path", e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
        </FormField>
        <FormField label="Student Model Path">
          <input value={extra["--student-model-path"] ?? ""} onChange={(e) => update("--student-model-path", e.currentTarget.value)} placeholder="optional — HF model ID or path, trains new student if empty" />
        </FormField>
        <FormField label="Temperature">
          <input value={extra["--temperature"] ?? "2.0"} onChange={(e) => update("--temperature", e.currentTarget.value)} />
        </FormField>
        <FormField label="Alpha">
          <input value={extra["--alpha"] ?? "0.5"} onChange={(e) => update("--alpha", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
