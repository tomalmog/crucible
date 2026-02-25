import { FormField } from "../../../components/shared/FormField";

interface GrpoTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function GrpoTrainForm({ extra, setExtra }: GrpoTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Group Relative Policy Optimization</h4>
      <div className="grid-2">
        <FormField label="GRPO Data Path">
          <input value={extra["--grpo-data-path"] ?? ""} onChange={(e) => update("--grpo-data-path", e.currentTarget.value)} placeholder="/path/to/prompts.jsonl" />
        </FormField>
        <FormField label="Base Model">
          <input value={extra["--base-model"] ?? ""} onChange={(e) => update("--base-model", e.currentTarget.value)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <input value={extra["--initial-weights-path"] ?? ""} onChange={(e) => update("--initial-weights-path", e.currentTarget.value)} placeholder="optional — .pt checkpoint to start from" />
        </FormField>
        <FormField label="Reward Function Path">
          <input value={extra["--reward-function-path"] ?? ""} onChange={(e) => update("--reward-function-path", e.currentTarget.value)} placeholder="optional — /path/to/reward.py" />
        </FormField>
        <FormField label="Group Size">
          <input type="number" value={extra["--group-size"] ?? "4"} onChange={(e) => update("--group-size", e.currentTarget.value)} />
        </FormField>
        <FormField label="KL Coefficient">
          <input value={extra["--kl-coeff"] ?? "0.1"} onChange={(e) => update("--kl-coeff", e.currentTarget.value)} />
        </FormField>
        <FormField label="Clip Range">
          <input value={extra["--clip-range"] ?? "0.2"} onChange={(e) => update("--clip-range", e.currentTarget.value)} />
        </FormField>
        <FormField label="Temperature">
          <input value={extra["--temperature"] ?? "1.0"} onChange={(e) => update("--temperature", e.currentTarget.value)} />
        </FormField>
      </div>
    </div>
  );
}
