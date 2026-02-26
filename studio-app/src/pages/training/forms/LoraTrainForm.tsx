import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface LoraTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function LoraTrainForm({ extra, setExtra }: LoraTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const baseModel = (extra["--base-model-path"] ?? "").trim();
  const hasHfModel = baseModel.length > 0 && !baseModel.endsWith(".pt") && !baseModel.endsWith(".bin") && !baseModel.startsWith("/") && !baseModel.startsWith(".");

  return (
    <div className="stack-sm">
      <h4>LoRA Training</h4>
      <div className="grid-2">
        <FormField label="Base Model" required>
          <PathInput value={extra["--base-model-path"] ?? ""} onChange={(v) => update("--base-model-path", v)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasHfModel ? "auto-loaded from base model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasHfModel ? "auto-loaded from base model" : "auto-detect from model"} disabled={hasHfModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
        <FormField label="LoRA Data Path" required>
          <PathInput value={extra["--lora-data-path"] ?? ""} onChange={(v) => update("--lora-data-path", v)} placeholder="/path/to/lora_data.jsonl" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
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
