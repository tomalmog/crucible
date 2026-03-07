import { useMemo, useRef, useState } from "react";
import { useForge } from "../../context/ForgeContext";

interface ModelSelectProps {
  value: string;
  onChange: (modelPath: string) => void;
  placeholder?: string;
}

/**
 * Searchable dropdown for selecting a registered model.
 * Shows model names in the dropdown; sets the active version's model path as value.
 * Uses the same CSS classes as DatasetSelect for consistent styling.
 */
export function ModelSelect({ value, onChange, placeholder = "select a registered model" }: ModelSelectProps) {
  const { modelGroups } = useForge();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Build path→name reverse lookup for displaying the selected model's name
  const pathToName = useMemo(() => {
    const map = new Map<string, string>();
    for (const g of modelGroups) {
      if (g.activeModelPath) map.set(g.activeModelPath, g.modelName);
    }
    return map;
  }, [modelGroups]);

  // Display the model name if the value matches a known path, otherwise show raw value
  const displayValue = value ? (pathToName.get(value) ?? value) : "";

  // Filter model groups by search text (match against model name)
  const query = open ? search : "";
  const filtered = modelGroups
    .filter((g) => g.activeModelPath)
    .filter((g) => g.modelName.toLowerCase().includes(query.toLowerCase()));

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setSearch(displayValue);
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => setOpen(false), 150);
  }

  function pick(modelPath: string): void {
    onChange(modelPath);
    setOpen(false);
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={open ? search : displayValue}
        onChange={(e) => setSearch(e.currentTarget.value)}
        placeholder={placeholder}
      />
      {open && filtered.length > 0 && (
        <ul className="dataset-select-dropdown">
          {filtered.map((g) => (
            <li key={g.modelName}>
              <button
                type="button"
                className="dataset-select-option"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(g.activeModelPath)}
              >
                {g.modelName}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
