import { FormField } from "../../../components/shared/FormField";

interface QloraTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function QloraTrainForm({ extra, setExtra }: QloraTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Quantized LoRA (QLoRA)</h4>
      <div className="grid-2">
        <FormField label="Training Data Path">
          <input value={extra["--qlora-data-path"] ?? ""} onChange={(e) => update("--qlora-data-path", e.currentTarget.value)} placeholder="/path/to/data.jsonl" />
        </FormField>
        <FormField label="Base Model">
          <input value={extra["--base-model-path"] ?? ""} onChange={(e) => update("--base-model-path", e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
        </FormField>
        <FormField label="Quantization Bits">
          <select value={extra["--quantization-bits"] ?? "4"} onChange={(e) => update("--quantization-bits", e.currentTarget.value)}>
            <option value="4">4-bit</option>
            <option value="8">8-bit</option>
          </select>
        </FormField>
        <FormField label="Quantization Type">
          <select value={extra["--qlora-type"] ?? "nf4"} onChange={(e) => update("--qlora-type", e.currentTarget.value)}>
            <option value="nf4">NF4</option>
            <option value="fp4">FP4</option>
          </select>
        </FormField>
        <FormField label="LoRA Rank">
          <input type="number" value={extra["--lora-rank"] ?? "8"} onChange={(e) => update("--lora-rank", e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Alpha">
          <input type="number" value={extra["--lora-alpha"] ?? "16"} onChange={(e) => update("--lora-alpha", e.currentTarget.value)} />
        </FormField>
        <FormField label="LoRA Dropout">
          <input value={extra["--lora-dropout"] ?? "0.0"} onChange={(e) => update("--lora-dropout", e.currentTarget.value)} />
        </FormField>
        <FormField label="Target Modules">
          <input value={extra["--lora-target-modules"] ?? "q_proj,v_proj"} onChange={(e) => update("--lora-target-modules", e.currentTarget.value)} placeholder="q_proj,v_proj or c_attn,c_proj for GPT-2" />
        </FormField>
      </div>
    </div>
  );
}
