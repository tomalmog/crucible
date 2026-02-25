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
        <FormField label="Base Model">
          <input value={extra["--base-model-path"] ?? ""} onChange={(e) => update("--base-model-path", e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
        </FormField>
        <FormField label="Tokenizer Path">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect from model" />
        </FormField>
        <FormField label="SFT Data Path (optional)">
          <input value={extra["--sft-data-path"] ?? ""} onChange={(e) => update("--sft-data-path", e.currentTarget.value)} placeholder="Uses dataset records if empty" />
        </FormField>
        <FormField label="LoRA Rank">
          <input type="number" value={extra["--lora-rank"] ?? "8"} onChange={(e) => update("--lora-rank", e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Alpha">
          <input type="number" value={extra["--lora-alpha"] ?? "16"} onChange={(e) => update("--lora-alpha", e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Dropout">
          <input type="number" value={extra["--lora-dropout"] ?? "0.0"} onChange={(e) => update("--lora-dropout", e.currentTarget.value)} step="0.01" />
        </FormField>
        <FormField label="Target Modules">
          <input value={extra["--lora-target-modules"] ?? "q_proj,v_proj"} onChange={(e) => update("--lora-target-modules", e.currentTarget.value)} placeholder="q_proj,v_proj or c_attn,c_proj for GPT-2" />
        </FormField>
      </div>
    </div>
  );
}
