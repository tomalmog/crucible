import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
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
 * Searchable dropdown for selecting a dataset.
 * Shows local datasets from the registry and, when a training cluster is
 * active, remote datasets from that cluster in a grouped "Remote" section.
 */
export function DatasetSelect({ value, onChange, placeholder = "dataset name" }: DatasetSelectProps) {
  const { datasets, dataRoot } = useCrucible();
  const cluster = useTrainingCluster();
  const [open, setOpen] = useState(false);
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [isLoadingRemote, setIsLoadingRemote] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Fetch remote datasets when a cluster is selected
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
    const lowerQuery = value.toLowerCase();
    for (const d of datasets) {
      if (d.name.toLowerCase().includes(lowerQuery)) {
        result.push({ label: d.name, value: d.name, section: "local" });
      }
    }
    for (const d of remoteDatasets) {
      if (d.name.toLowerCase().includes(lowerQuery)) {
        result.push({ label: d.name, value: d.name, section: "remote" });
      }
    }
    return result;
  }, [datasets, remoteDatasets, value]);

  const localOptions = options.filter((o) => o.section === "local");
  const remoteOptions = options.filter((o) => o.section === "remote");
  const hasResults = localOptions.length > 0 || remoteOptions.length > 0;

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

  const showSections = remoteOptions.length > 0;

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <input
        value={value}
        onChange={(e) => onChange(e.currentTarget.value)}
        placeholder={placeholder}
      />
      {open && (hasResults || isLoadingRemote) && (
        <ul className="dataset-select-dropdown">
          {localOptions.length > 0 && (
            <>
              {(showSections || isLoadingRemote) && (
                <li className="dataset-select-header">Local</li>
              )}
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
        </ul>
      )}
    </div>
  );
}
