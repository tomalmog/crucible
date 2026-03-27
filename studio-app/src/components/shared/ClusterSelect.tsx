import { useRef, useState } from "react";
import type { ClusterConfig } from "../../types/remote";

interface ClusterSelectProps {
  clusters: ClusterConfig[];
  value: string;
  onChange: (name: string) => void;
}

export function ClusterSelect({ clusters, value, onChange }: ClusterSelectProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const filtered = clusters.filter((c) =>
    c.name.toLowerCase().includes(query.toLowerCase()),
  );

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setQuery("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => setOpen(false), 150);
  }

  function pick(name: string): void {
    onChange(name);
    setQuery("");
    setOpen(false);
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={open ? query : value}
        onChange={(e) => setQuery(e.currentTarget.value)}
        placeholder="Search clusters…"
        readOnly={!open}
        style={{ width: 100, fontSize: "0.75rem" }}
      />
      {open && filtered.length > 0 && (
        <ul className="dataset-select-dropdown">
          {filtered.map((c) => (
            <li key={c.name}>
              <button
                type="button"
                className="dataset-select-option"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(c.name)}
              >
                {c.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
