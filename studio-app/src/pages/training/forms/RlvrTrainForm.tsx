import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";

interface RlvrTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function RlvrTrainForm({ extra, setExtra }: RlvrTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>RL with Verifiable Rewards</h4>
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} />
        </FormField>
        <FormField label="Verifier Type">
          <select value={extra["--verifier-type"] ?? "code"} onChange={(e) => update("--verifier-type", e.currentTarget.value)}>
            <option value="code">Code</option>
            <option value="math">Math</option>
          </select>
        </FormField>
        <FormField label="Max Verification Attempts">
          <input type="number" value={extra["--max-attempts"] ?? "3"} onChange={(e) => update("--max-attempts", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reward (Correct)">
          <input value={extra["--reward-correct"] ?? "1.0"} onChange={(e) => update("--reward-correct", e.currentTarget.value)} />
        </FormField>
        <FormField label="Reward (Incorrect)">
          <input value={extra["--reward-incorrect"] ?? "-0.5"} onChange={(e) => update("--reward-incorrect", e.currentTarget.value)} />
        </FormField>
      </div>
    </div>
  );
}
