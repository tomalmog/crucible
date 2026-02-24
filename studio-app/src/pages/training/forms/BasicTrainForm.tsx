import { FormField } from "../../../components/shared/FormField";

interface BasicTrainFormProps {
  extra: Record<string, string>;
  setExtra: (extra: Record<string, string>) => void;
}

export function BasicTrainForm({ extra, setExtra }: BasicTrainFormProps) {
  function update(key: string, value: string) {
    setExtra({ ...extra, [key]: value });
  }

  return (
    <div>
      <h4 className="panel-title">Basic Training</h4>
      <p className="text-tertiary page-description">
        Standard supervised training from dataset. Configure shared parameters below.
      </p>
      <div className="gap-top-sm">
        <FormField label="Tokenizer Path (optional)">
          <input value={extra["--tokenizer-path"] ?? ""} onChange={(e) => update("--tokenizer-path", e.currentTarget.value)} placeholder="auto-detect" />
        </FormField>
      </div>
    </div>
  );
}
