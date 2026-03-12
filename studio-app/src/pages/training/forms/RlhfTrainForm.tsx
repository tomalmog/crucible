import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
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
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Policy Model" required>
          <ModelSelect value={extra["--policy-model-path"] ?? ""} onChange={(v) => update("--policy-model-path", v)} />
        </FormField>
        <FormField label="Reward Model">
          <ModelSelect value={extra["--reward-model-path"] ?? ""} onChange={(v) => update("--reward-model-path", v)} placeholder="optional — or train from preference data" />
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
