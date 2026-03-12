import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Loader2, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useTrainingCluster } from "../../context/TrainingClusterContext";
import { listRemoteDatasets } from "../../api/remoteApi";
import type { RemoteDatasetInfo } from "../../types/remote";

interface DatasetSelectProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

interface DatasetOption {
  label: string;
  value: string;
  section: "local" | "remote";
}

/**
 * Searchable dropdown for selecting a dataset from the registry.
 * Only allows picking from local/remote datasets — no free text.
 */
export function DatasetSelect({ value, onChange, placeholder = "Select a dataset" }: DatasetSelectProps) {
  const { datasets, dataRoot } = useCrucible();
  const cluster = useTrainingCluster();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [isLoadingRemote, setIsLoadingRemote] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (!dataRoot || !cluster) {
      setRemoteDatasets([]);
      return;
    }
    setIsLoadingRemote(true);
    listRemoteDatasets(dataRoot, cluster)
      .then(setRemoteDatasets)
      .catch(() => setRemoteDatasets([]))
      .finally(() => setIsLoadingRemote(false));
  }, [dataRoot, cluster]);

  const options = useMemo(() => {
    const result: DatasetOption[] = [];
    const q = search.toLowerCase();
    for (const d of datasets) {
      if (d.name.toLowerCase().includes(q)) {
        result.push({ label: d.name, value: d.name, section: "local" });
      }
    }
    for (const d of remoteDatasets) {
      if (d.name.toLowerCase().includes(q)) {
        result.push({ label: d.name, value: d.name, section: "remote" });
      }
    }
    return result;
  }, [datasets, remoteDatasets, search]);

  const localOptions = options.filter((o) => o.section === "local");
  const remoteOptions = options.filter((o) => o.section === "remote");
  const hasResults = localOptions.length > 0 || remoteOptions.length > 0;

  function handleFocus() {
    clearTimeout(blurTimeout.current);
    setSearch("");
    setOpen(true);
  }

  function handleBlur() {
    blurTimeout.current = setTimeout(() => {
      setOpen(false);
      setSearch("");
    }, 150);
  }

  function pick(name: string) {
    onChange(name);
    setOpen(false);
    setSearch("");
  }

  function clear() {
    onChange("");
    setSearch("");
  }

  function renderOptions(items: DatasetOption[]) {
    return items.map((o) => (
      <li key={`${o.section}-${o.value}`}>
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
          value={open ? search : value}
          onChange={(e) => setSearch(e.currentTarget.value)}
          placeholder={value || placeholder}
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
      {open && (hasResults || isLoadingRemote) && (
        <ul className="dataset-select-dropdown">
          {localOptions.length > 0 && (
            <>
              <li className="dataset-select-header">Local</li>
              {renderOptions(localOptions)}
            </>
          )}
          {isLoadingRemote ? (
            <li className="dataset-select-header" style={{ display: "flex", alignItems: "center", gap: 6 }}>
              Remote <Loader2 size={12} className="spin" />
            </li>
          ) : remoteOptions.length > 0 ? (
            <>
              <li className="dataset-select-header">Remote</li>
              {renderOptions(remoteOptions)}
            </>
          ) : null}
          {!hasResults && !isLoadingRemote && (
            <li className="dataset-select-empty">No datasets found</li>
          )}
        </ul>
      )}
    </div>
  );
}
