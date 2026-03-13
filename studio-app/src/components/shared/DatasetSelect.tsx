import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Loader2, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useTrainingCluster } from "../../context/TrainingClusterContext";
import { listClusters, listRemoteDatasets } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
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
 * Always shows both local and remote datasets when clusters exist.
 */
export function DatasetSelect({ value, onChange, placeholder = "Select a dataset" }: DatasetSelectProps) {
  const { datasets, dataRoot } = useCrucible();
  const { cluster: contextCluster, onDatasetLocationChanged } = useTrainingCluster();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [isLoadingRemote, setIsLoadingRemote] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Always fetch clusters on mount so we can show remote datasets
  useEffect(() => {
    if (!dataRoot) return;
    listClusters(dataRoot)
      .then(setClusters)
      .catch(() => setClusters([]));
  }, [dataRoot]);

  // Fetch remote datasets: use context cluster if set, otherwise first available
  const effectiveCluster = contextCluster || clusters[0]?.name || "";

  useEffect(() => {
    if (!dataRoot || !effectiveCluster) {
      setRemoteDatasets([]);
      return;
    }
    setIsLoadingRemote(true);
    listRemoteDatasets(dataRoot, effectiveCluster)
      .then(setRemoteDatasets)
      .catch(() => setRemoteDatasets([]))
      .finally(() => setIsLoadingRemote(false));
  }, [dataRoot, effectiveCluster]);

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

  function pick(name: string, section: "local" | "remote") {
    onChange(name);
    if (onDatasetLocationChanged) {
      onDatasetLocationChanged(section === "remote", effectiveCluster);
    }
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
          onClick={() => pick(o.value, o.section)}
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
          onClick={() => { if (!open) { setSearch(""); setOpen(true); } }}
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
