import { FormField } from "../../../components/shared/FormField";

interface LoraTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function LoraTrainForm({ extra, setExtra }: LoraTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>LoRA Training</h4>
      <div className="grid-2">
        <FormField label="Base Model Path">
          <input value={extra["--base-model-path"] ?? ""} onChange={(e) => update("--base-model-path", e.currentTarget.value)} placeholder="/path/to/base_model.pt" />
        </FormField>
        <FormField label="LoRA Rank">
          <input type="number" value={extra["--lora-rank"] ?? "8"} onChange={(e) => update("--lora-rank", e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Alpha">
          <input type="number" value={extra["--lora-alpha"] ?? "16"} onChange={(e) => update("--lora-alpha", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
