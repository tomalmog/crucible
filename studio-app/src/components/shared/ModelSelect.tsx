import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";

interface ModelSelectProps {
  value: string;
  onChange: (modelPath: string) => void;
  placeholder?: string;
  remoteOnly?: boolean;
  localOnly?: boolean;
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
export function ModelSelect({ value, onChange, placeholder = "Select a model", remoteOnly = false, localOnly = false }: ModelSelectProps) {
  const { models, dataRoot } = useCrucible();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);

  useEffect(() => {
    if (dataRoot) listClusters(dataRoot).then(setClusters).catch(() => {});
  }, [dataRoot]);

  const hostToCluster = useMemo(() => {
    const map = new Map<string, string>();
    for (const c of clusters) map.set(c.host, c.name);
    return map;
  }, [clusters]);

  const pathToDisplay = useMemo(() => {
    const map = new Map<string, string>();
    for (const m of models) {
      if (m.modelPath) map.set(m.modelPath, `${m.modelName} — Local`);
      if (m.remotePath) {
        const label = hostToCluster.get(m.remoteHost) || m.remoteHost;
        map.set(m.remotePath, `${m.modelName} — ${label}`);
      }
    }
    return map;
  }, [models, hostToCluster]);

  const displayValue = value ? (pathToDisplay.get(value) ?? value) : "";

  const options = useMemo(() => {
    const result: ModelOption[] = [];
    const q = search.toLowerCase();
    for (const m of models) {
      if (!m.modelName.toLowerCase().includes(q)) continue;
      if (!remoteOnly && m.hasLocal && m.modelPath) {
        result.push({ label: m.modelName, value: m.modelPath, section: "local" });
      }
      if (!localOnly && m.hasRemote && m.remotePath) {
        result.push({ label: m.modelName, value: m.remotePath, section: "remote" });
      }
    }
    return result;
  }, [models, search, remoteOnly, localOnly]);

  // Group remote models by cluster hostname
  const remoteGroups = useMemo(() => {
    const groups = new Map<string, ModelOption[]>();
    if (localOnly) return groups;
    for (const m of models) {
      if (!m.hasRemote || !m.remotePath) continue;
      const q = search.toLowerCase();
      if (!m.modelName.toLowerCase().includes(q)) continue;
      const label = hostToCluster.get(m.remoteHost) || m.remoteHost;
      const list = groups.get(label) || [];
      list.push({ label: m.modelName, value: m.remotePath, section: "remote" });
      groups.set(label, list);
    }
    return groups;
  }, [models, search, localOnly, hostToCluster]);

  const localOptions = options.filter((o) => o.section === "local");
  const hasRemote = remoteGroups.size > 0;
  const hasResults = localOptions.length > 0 || hasRemote;

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
          onClick={() => { if (!open) { setSearch(""); setOpen(true); } }}
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
          {Array.from(remoteGroups.entries()).map(([host, items]) => (
            <React.Fragment key={host}>
              <li className="dataset-select-header">{host}</li>
              {renderOptions(items)}
            </React.Fragment>
          ))}
          {!hasResults && (
            <li className="dataset-select-empty">No models found</li>
          )}
        </ul>
      )}
    </div>
  );
}
