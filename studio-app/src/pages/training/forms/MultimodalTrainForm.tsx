import { FormField } from "../../../components/shared/FormField";
import { PathInput } from "../../../components/shared/PathInput";

interface MultimodalTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function MultimodalTrainForm({ extra, setExtra }: MultimodalTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div className="stack-sm">
      <h4>Multimodal Fine-Tuning</h4>
      <div className="grid-2">
        <FormField label="Multimodal Data Path" required>
          <PathInput value={extra["--multimodal-data-path"] ?? ""} onChange={(v) => update("--multimodal-data-path", v)} placeholder="/path/to/image_text.jsonl" filters={[{ name: "JSONL", extensions: ["jsonl"] }]} />
        </FormField>
        <FormField label="Base Model" required>
          <PathInput value={extra["--base-model"] ?? ""} onChange={(v) => update("--base-model", v)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <PathInput value={extra["--initial-weights-path"] ?? ""} onChange={(v) => update("--initial-weights-path", v)} placeholder="optional — .pt checkpoint to start from" filters={[{ name: "Checkpoint", extensions: ["pt"] }]} />
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
