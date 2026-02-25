import { FormField } from "../../../components/shared/FormField";

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
        <FormField label="Policy Model Path">
          <input value={extra["--policy-model-path"] ?? ""} onChange={(e) => update("--policy-model-path", e.currentTarget.value)} placeholder="gpt2, meta-llama/Llama-2-7b, or /path/to/model.pt" />
        </FormField>
        <FormField label="Reward Model Path">
          <input value={extra["--reward-model-path"] ?? ""} onChange={(e) => update("--reward-model-path", e.currentTarget.value)} placeholder="optional — or train from preference data" />
        </FormField>
        <FormField label="Preference Data Path">
          <input value={extra["--preference-data-path"] ?? ""} onChange={(e) => update("--preference-data-path", e.currentTarget.value)} placeholder="/path/to/preferences.jsonl (for reward training)" />
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
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
