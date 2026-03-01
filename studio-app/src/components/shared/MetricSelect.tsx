import { useRef, useState } from "react";

const METRIC_OPTIONS = [
  "validation_loss",
  "train_loss",
  "mean_reward",
  "accuracy",
  "f1_score",
  "perplexity",
  "bleu",
  "rouge_l",
];

interface MetricSelectProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function MetricSelect({ value, onChange, placeholder = "metric name" }: MetricSelectProps) {
  const [open, setOpen] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const filtered = METRIC_OPTIONS.filter((name) =>
    name.toLowerCase().includes(value.toLowerCase()),
  );

  function handleFocus() {
    clearTimeout(blurTimeout.current);
    setOpen(true);
  }

  function handleBlur() {
    blurTimeout.current = setTimeout(() => setOpen(false), 150);
  }

  function pick(name: string) {
    onChange(name);
    setOpen(false);
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={value}
        onChange={(e) => onChange(e.currentTarget.value)}
        placeholder={placeholder}
      />
      {open && filtered.length > 0 && (
        <ul className="dataset-select-dropdown">
          {filtered.map((name) => (
            <li key={name}>
              <button
                type="button"
                className="dataset-select-option"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(name)}
              >
                {name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
