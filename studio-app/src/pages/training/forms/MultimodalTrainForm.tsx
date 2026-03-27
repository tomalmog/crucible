import { DatasetSelect } from "../../../components/shared/DatasetSelect";
import { FormField } from "../../../components/shared/FormField";
import { ModelSelect } from "../../../components/shared/ModelSelect";
import { PathInput } from "../../../components/shared/PathInput";

interface MultimodalTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function MultimodalTrainForm({ extra, setExtra }: MultimodalTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  const hasModel = (extra["--base-model"] ?? "").trim().length > 0;

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
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to resume from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
        </FormField>
        <FormField label="Tokenizer Path" hint={hasModel ? "auto-loaded from model" : undefined}>
          <PathInput value={extra["--tokenizer-path"] ?? ""} onChange={(v) => update("--tokenizer-path", v)} placeholder={hasModel ? "auto-loaded from model" : "auto-detect"} disabled={hasModel && !(extra["--tokenizer-path"] ?? "").trim()} filters={[{ name: "JSON", extensions: ["json"] }]} />
        </FormField>
        <FormField label="Image Encoder">
          <input value={extra["--image-encoder"] ?? "clip-vit-base"} onChange={(e) => update("--image-encoder", e.currentTarget.value)} />
        </FormField>
        <FormField label="Image Size">
          <input type="number" value={extra["--image-size"] ?? "224"} onChange={(e) => update("--image-size", e.currentTarget.value)} />
        </FormField>
        <FormField label="Projection Dimension">
          <input type="number" value={extra["--projection-dim"] ?? "512"} onChange={(e) => update("--projection-dim", e.currentTarget.value)} />
        </FormField>
      </div>
    </div>
  );
}
