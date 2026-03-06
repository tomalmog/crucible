import { useRef, useState } from "react";
import { useForge } from "../../context/ForgeContext";

interface DatasetSelectProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function DatasetSelect({ value, onChange, placeholder = "dataset name" }: DatasetSelectProps) {
  const { datasets } = useForge();
  const [open, setOpen] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const filtered = datasets
    .map((d) => d.name)
    .filter((name) => name.toLowerCase().includes(value.toLowerCase()));

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
