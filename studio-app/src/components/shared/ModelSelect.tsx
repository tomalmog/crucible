import { useMemo, useRef, useState } from "react";
import { ChevronDown, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";

interface ModelSelectProps {
  value: string;
  onChange: (modelPath: string) => void;
  placeholder?: string;
  remoteOnly?: boolean;
}

interface ModelOption {
  label: string;
  value: string;
  section: "local" | "remote";
}

/**
 * Searchable dropdown for selecting a registered model.
 * Only allows picking from the model registry — no free text.
 */
export function ModelSelect({ value, onChange, placeholder = "Select a model", remoteOnly = false }: ModelSelectProps) {
  const { models } = useCrucible();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  const pathToName = useMemo(() => {
    const map = new Map<string, string>();
    for (const m of models) {
      if (m.modelPath) map.set(m.modelPath, m.modelName);
      if (m.remotePath) {
        const host = m.remoteHost.split(".")[0];
        map.set(m.remotePath, `${m.modelName} (${host})`);
      }
    }
    return map;
  }, [models]);

  const displayValue = value ? (pathToName.get(value) ?? value) : "";

  const options = useMemo(() => {
    const result: ModelOption[] = [];
    const q = search.toLowerCase();
    for (const m of models) {
      if (!m.modelName.toLowerCase().includes(q)) continue;
      if (!remoteOnly && m.hasLocal && m.modelPath) {
        result.push({ label: m.modelName, value: m.modelPath, section: "local" });
      }
      if (m.hasRemote && m.remotePath) {
        const host = m.remoteHost.split(".")[0];
        result.push({ label: `${m.modelName} (${host})`, value: m.remotePath, section: "remote" });
      }
    }
    return result;
  }, [models, search, remoteOnly]);

  const localOptions = options.filter((o) => o.section === "local");
  const remoteOptions = options.filter((o) => o.section === "remote");
  const hasResults = localOptions.length > 0 || remoteOptions.length > 0;

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setSearch("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => {
      setOpen(false);
      setSearch("");
    }, 150);
  }

  function pick(modelPath: string): void {
    onChange(modelPath);
    setOpen(false);
    setSearch("");
  }

  function clear(): void {
    onChange("");
    setSearch("");
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
      <div className="dataset-select-input-wrap">
        <input
          value={open ? search : displayValue}
          onChange={(e) => setSearch(e.currentTarget.value)}
          placeholder={displayValue || placeholder}
          readOnly={!open}
        />
        {value && !open ? (
          <button type="button" className="dataset-select-clear" onMouseDown={(e) => e.preventDefault()} onClick={clear}>
            <X size={14} />
          </button>
        ) : (
          <ChevronDown size={14} className="dataset-select-chevron" />
        )}
      </div>
      {open && (
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
          {!hasResults && (
            <li className="dataset-select-empty">No models found</li>
          )}
        </ul>
      )}
    </div>
  );
}
