import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

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
      <div className="grid-2">
        <FormField label="Dataset" required>
          <DatasetSelect value={extra["--dataset"] ?? ""} onChange={(v) => update("--dataset", v)} />
        </FormField>
        <FormField label="Base Model" required>
          <ModelSelect value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to start from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Reward Function Path">
          <PathInput value={extra["--reward-function-path"] ?? ""} onChange={(v) => update("--reward-function-path", v)} placeholder="optional — /path/to/reward.py" filters={[{ name: "Python", extensions: ["py"] }]} />
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
