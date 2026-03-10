import { useMemo, useRef, useState } from "react";
import { useCrucible } from "../../context/CrucibleContext";

interface ModelSelectProps {
  value: string;
  onChange: (modelPath: string) => void;
  placeholder?: string;
}

interface ModelOption {
  label: string;
  value: string;
  section: "local" | "remote";
}

/**
 * Searchable dropdown for selecting a registered model.
 * Shows local and remote models in grouped sections.
 * Local models pass activeModelPath; remote models pass activeRemotePath.
 */
export function ModelSelect({ value, onChange, placeholder = "select a registered model" }: ModelSelectProps) {
  const { modelGroups } = useCrucible();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const pathToName = useMemo(() => {
    const map = new Map<string, string>();
    for (const g of modelGroups) {
      if (g.activeModelPath) map.set(g.activeModelPath, g.modelName);
      if (g.activeRemotePath) {
        const host = g.activeRemoteHost.split(".")[0];
        map.set(g.activeRemotePath, `${g.modelName} (${host})`);
      }
    }
    return map;
  }, [modelGroups]);

  const displayValue = value ? (pathToName.get(value) ?? value) : "";

  const query = open ? search : "";
  const options = useMemo(() => {
    const result: ModelOption[] = [];
    const lowerQuery = query.toLowerCase();
    for (const g of modelGroups) {
      if (!g.modelName.toLowerCase().includes(lowerQuery)) continue;
      if (g.hasLocal && g.activeModelPath) {
        result.push({ label: g.modelName, value: g.activeModelPath, section: "local" });
      }
      if (g.hasRemote && g.activeRemotePath) {
        const host = g.activeRemoteHost.split(".")[0];
        result.push({ label: `${g.modelName} (${host})`, value: g.activeRemotePath, section: "remote" });
      }
    }
    return result;
  }, [modelGroups, query]);

  const localOptions = options.filter((o) => o.section === "local");
  const remoteOptions = options.filter((o) => o.section === "remote");
  const hasResults = localOptions.length > 0 || remoteOptions.length > 0;

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

  function renderOptions(items: ModelOption[]) {
    return items.map((o) => (
      <li key={o.value}>
        <button
          type="button"
          className="dataset-select-option"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => pick(o.value)}
        >
          {o.label}
        </button>
      </li>
    ));
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={open ? search : displayValue}
        onChange={(e) => {
          setSearch(e.currentTarget.value);
          if (!open) setOpen(true);
        }}
        placeholder={placeholder}
      />
      {open && hasResults && (
        <ul className="dataset-select-dropdown">
          {localOptions.length > 0 && (
            <>
              <li className="dataset-select-header">Local</li>
              {renderOptions(localOptions)}
            </>
          )}
          {remoteOptions.length > 0 && (
            <>
              <li className="dataset-select-header">Remote</li>
              {renderOptions(remoteOptions)}
            </>
          )}
        </ul>
      )}
    </div>
  );
}
