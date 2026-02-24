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
        <FormField label="Reward Model Path">
          <input value={extra["--reward-model-path"] ?? ""} onChange={(e) => update("--reward-model-path", e.currentTarget.value)} placeholder="/path/to/reward_model.pt" />
        </FormField>
        <FormField label="PPO Epochs">
          <input type="number" value={extra["--ppo-epochs"] ?? "4"} onChange={(e) => update("--ppo-epochs", e.currentTarget.value)} />
        </FormField>
        <FormField label="KL Coefficient">
          <input value={extra["--kl-coeff"] ?? "0.02"} onChange={(e) => update("--kl-coeff", e.currentTarget.value)} />
        </FormField>
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
