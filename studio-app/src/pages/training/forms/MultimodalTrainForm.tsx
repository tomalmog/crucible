import { FormField } from "../../../components/shared/FormField";

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
        <FormField label="Multimodal Data Path">
          <input value={extra["--multimodal-data-path"] ?? ""} onChange={(e) => update("--multimodal-data-path", e.currentTarget.value)} placeholder="/path/to/image_text.jsonl" />
        </FormField>
        <FormField label="Base Model">
          <input value={extra["--base-model"] ?? ""} onChange={(e) => update("--base-model", e.currentTarget.value)} placeholder="HuggingFace model ID (e.g. gpt2, meta-llama/Llama-2-7b)" />
        </FormField>
        <FormField label="Initial Weights">
          <input value={extra["--initial-weights-path"] ?? ""} onChange={(e) => update("--initial-weights-path", e.currentTarget.value)} placeholder="optional — .pt checkpoint to start from" />
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
