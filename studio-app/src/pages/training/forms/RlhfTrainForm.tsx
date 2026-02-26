import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface RlhfTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function RlhfTrainForm({ extra, setExtra }: RlhfTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>RLHF Training</h4>
      <div className="grid-2">
        <FormField label="Policy Model Path" required>
          <PathInput value={extra["--policy-model-path"] ?? ""} onChange={(v) => update("--policy-model-path", v)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Reward Model Path">
          <PathInput value={extra["--reward-model-path"] ?? ""} onChange={(v) => update("--reward-model-path", v)} placeholder="optional — or train from preference data" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Preference Data Path">
          <PathInput value={extra["--preference-data-path"] ?? ""} onChange={(v) => update("--preference-data-path", v)} placeholder="/path/to/preferences.jsonl (for reward training)" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
        </FormField>
        <FormField label="Train Reward Model">
          <select value={extra["--train-reward-model"] ?? ""} onChange={(e) => update("--train-reward-model", e.currentTarget.value)}>
            <option value="">No</option>
            <option value="true">Yes — train from preference data</option>
          </select>
        </FormField>
        <FormField label="PPO Epochs">
          <input type="number" value={extra["--ppo-epochs"] ?? "4"} onChange={(e) => update("--ppo-epochs", e.currentTarget.value)} />
        </FormField>
        <FormField label="Clip Epsilon">
          <input value={extra["--clip-epsilon"] ?? "0.2"} onChange={(e) => update("--clip-epsilon", e.currentTarget.value)} />
        </FormField>
        <FormField label="Entropy Coefficient">
          <input value={extra["--entropy-coeff"] ?? "0.01"} onChange={(e) => update("--entropy-coeff", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path">
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder="auto-detect" filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
      </div>
    </div>
  );
}
