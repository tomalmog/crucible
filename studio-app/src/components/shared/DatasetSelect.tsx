import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Loader2, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useTrainingCluster } from "../../context/TrainingClusterContext";
import { listClusters, listRemoteDatasets } from "../../api/remoteApi";
import type { ClusterConfig, RemoteDatasetInfo } from "../../types/remote";

interface DatasetSelectProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

interface DatasetOption {
  label: string;
  value: string;
  section: "local" | "remote";
  cluster?: string;
}

interface ClusterDatasets {
  cluster: string;
  shortName: string;
  datasets: RemoteDatasetInfo[];
}

/**
 * Searchable dropdown for selecting a dataset from the registry.
 * Shows local datasets and remote datasets from ALL registered clusters.
 */
export function DatasetSelect({ value, onChange, placeholder = "Select a dataset" }: DatasetSelectProps) {
  const { datasets, dataRoot } = useCrucible();
  const { onDatasetLocationChanged } = useTrainingCluster();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [pickedSection, setPickedSection] = useState<"local" | "remote">("local");
  const [pickedCluster, setPickedCluster] = useState("");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [clusterDatasets, setClusterDatasets] = useState<ClusterDatasets[]>([]);
  const [isLoadingRemote, setIsLoadingRemote] = useState(false);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Fetch clusters on mount
  useEffect(() => {
    if (!dataRoot) return;
    listClusters(dataRoot)
      .then(setClusters)
      .catch(() => setClusters([]));
  }, [dataRoot]);

  // Fetch remote datasets from ALL clusters
  useEffect(() => {
    if (!dataRoot || clusters.length === 0) {
      setClusterDatasets([]);
      return;
    }
    setIsLoadingRemote(true);
    Promise.all(
      clusters.map(async (c) => {
        const ds = await listRemoteDatasets(dataRoot, c.name).catch(() => [] as RemoteDatasetInfo[]);
        return { cluster: c.name, shortName: c.name.split(".")[0] || c.name, datasets: ds };
      }),
    )
      .then(setClusterDatasets)
      .catch(() => setClusterDatasets([]))
      .finally(() => setIsLoadingRemote(false));
  }, [dataRoot, clusters]);

  const options = useMemo(() => {
    const result: DatasetOption[] = [];
    const q = search.toLowerCase();
    for (const d of datasets) {
      if (d.name.toLowerCase().includes(q)) {
        result.push({ label: d.name, value: d.name, section: "local" });
      }
    }
    for (const cd of clusterDatasets) {
      for (const d of cd.datasets) {
        if (d.name.toLowerCase().includes(q)) {
          result.push({ label: d.name, value: d.name, section: "remote", cluster: cd.cluster });
        }
      }
    }
    return result;
  }, [datasets, clusterDatasets, search]);

  // Map dataset name → display label (with location suffix)
  const valueToDisplay = useMemo(() => {
    const map = new Map<string, string>();
    for (const d of datasets) {
      map.set(`local::${d.name}`, `${d.name} — Local`);
    }
    for (const cd of clusterDatasets) {
      for (const d of cd.datasets) {
        map.set(`remote:${cd.cluster}:${d.name}`, `${d.name} — ${cd.shortName}`);
      }
    }
    return map;
  }, [datasets, clusterDatasets]);

  // Group remote options by cluster
  const remoteGroups = useMemo(() => {
    const groups = new Map<string, { shortName: string; items: DatasetOption[] }>();
    for (const cd of clusterDatasets) {
      const q = search.toLowerCase();
      const filtered = cd.datasets.filter((d) => d.name.toLowerCase().includes(q));
      if (filtered.length > 0) {
        groups.set(cd.cluster, {
          shortName: cd.shortName,
          items: filtered.map((d) => ({ label: d.name, value: d.name, section: "remote" as const, cluster: cd.cluster })),
        });
      }
    }
    return groups;
  }, [clusterDatasets, search]);

  const localOptions = options.filter((o) => o.section === "local");
  const hasRemote = remoteGroups.size > 0;
  const hasResults = localOptions.length > 0 || hasRemote;

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

  function pick(name: string, section: "local" | "remote", cluster?: string) {
    onChange(name);
    setPickedSection(section);
    setPickedCluster(cluster || "");
    if (onDatasetLocationChanged) {
      onDatasetLocationChanged(section === "remote", cluster || "");
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
      <li key={`${o.section}-${o.cluster || "local"}-${o.value}`}>
        <button
          type="button"
          className="dataset-select-option"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => pick(o.value, o.section, o.cluster)}
        >
          {o.label}
        </button>
      </li>
    ));
  }

  const displayKey = pickedSection === "local" ? `local::${value}` : `remote:${pickedCluster}:${value}`;

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <div className="dataset-select-input-wrap">
        <input
          value={open ? search : (value ? (valueToDisplay.get(displayKey) || value) : "")}
          onChange={(e) => setSearch(e.currentTarget.value)}
          onClick={() => { if (!open) { setSearch(""); setOpen(true); } }}
          placeholder={value ? (valueToDisplay.get(displayKey) || value) : placeholder}
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
              Loading clusters… <Loader2 size={12} className="spin" />
            </li>
          ) : (
            Array.from(remoteGroups.entries()).map(([cluster, { shortName, items }]) => (
              <React.Fragment key={cluster}>
                <li className="dataset-select-header">{shortName}</li>
                {renderOptions(items)}
              </React.Fragment>
            ))
          )}
          {!hasResults && !isLoadingRemote && (
            <li className="dataset-select-empty">No datasets found</li>
          )}
        </ul>
      )}
    </div>
  );
}
