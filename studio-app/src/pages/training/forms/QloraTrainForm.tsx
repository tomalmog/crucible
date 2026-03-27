import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface QloraTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function QloraTrainForm({ extra, setExtra }: QloraTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasModel = (extra["--base-model-path"] ?? "").trim().length > 0;

  return (
    <div className="stack-sm">
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model-path"] ?? ""} onChange={(v) => update("--base-model-path", v)} />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to resume from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasModel ? "auto-loaded from model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasModel ? "auto-loaded from model" : "auto-detect"} disabled={hasModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
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
        <FormField label="Double Quantization" hint="double-quantize QLoRA quantization constants">
          <label className="toggle-row">
            <input type="checkbox" checked={(extra["--double-quantize"] ?? "true") === "true"} onChange={(e) => update("--double-quantize", e.currentTarget.checked ? "true" : "false")} />
            <span>{(extra["--double-quantize"] ?? "true") === "true" ? "Enabled" : "Disabled"}</span>
          </label>
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
