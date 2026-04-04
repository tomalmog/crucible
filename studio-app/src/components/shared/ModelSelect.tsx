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
  /** Unique key: "local::<name>" or "remote::<name>::<host>" */
  key: string;
  /** Actual path passed to onChange */
  path: string;
  section: "local" | "remote";
}

/**
 * Searchable dropdown for selecting a registered model.
 *
 * Internally tracks selection by model name (unique) rather than path,
 * because multiple models can share the same output path.
 * The onChange callback still receives the file path for CLI use.
 */
export function ModelSelect({ value, onChange, placeholder = "Select a model", remoteOnly = false, localOnly = false }: ModelSelectProps) {
  const { models, dataRoot } = useCrucible();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [selectedKey, setSelectedKey] = useState("");
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

  // Build options keyed by name (unique) instead of path (can collide).
  // Remote models are only shown if their host belongs to a registered cluster.
  const { allOptions, keyToDisplay, pathToKey } = useMemo(() => {
    const opts: ModelOption[] = [];
    const display = new Map<string, string>();
    const p2k = new Map<string, string>();
    const knownHosts = new Set(hostToCluster.keys());
    for (const m of models) {
      if (!remoteOnly && m.hasLocal && m.modelPath) {
        const key = `local::${m.modelName}`;
        opts.push({ label: m.modelName, key, path: m.modelPath, section: "local" });
        display.set(key, `${m.modelName} — Local`);
        // Only set path→key if not already set (avoids collision overwrite)
        if (!p2k.has(m.modelPath)) p2k.set(m.modelPath, key);
      }
      if (!localOnly && m.hasRemote && m.remotePath && knownHosts.has(m.remoteHost)) {
        const clusterLabel = hostToCluster.get(m.remoteHost) || m.remoteHost;
        const key = `remote::${m.modelName}::${m.remoteHost}`;
        opts.push({ label: m.modelName, key, path: m.remotePath, section: "remote" });
        display.set(key, `${m.modelName} — ${clusterLabel}`);
        if (!p2k.has(m.remotePath)) p2k.set(m.remotePath, key);
      }
    }
    return { allOptions: opts, keyToDisplay: display, pathToKey: p2k };
  }, [models, remoteOnly, localOnly, hostToCluster]);

  // Sync selectedKey from the value prop (path) on mount or external changes
  useEffect(() => {
    if (!value) { setSelectedKey(""); return; }
    const mapped = pathToKey.get(value);
    if (mapped && !selectedKey) setSelectedKey(mapped);
  }, [value, pathToKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const displayValue = selectedKey ? (keyToDisplay.get(selectedKey) ?? "") : "";

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return q ? allOptions.filter((o) => o.label.toLowerCase().includes(q)) : allOptions;
  }, [allOptions, search]);

  const localOptions = filtered.filter((o) => o.section === "local");
  const remoteGroups = useMemo(() => {
    const groups = new Map<string, ModelOption[]>();
    for (const o of filtered) {
      if (o.section !== "remote") continue;
      const clusterLabel = o.key.split("::")[2] || "";
      const label = hostToCluster.get(clusterLabel) || clusterLabel;
      const list = groups.get(label) || [];
      list.push(o);
      groups.set(label, list);
    }
    return groups;
  }, [filtered, hostToCluster]);

  const hasResults = localOptions.length > 0 || remoteGroups.size > 0;

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setSearch("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => { setOpen(false); setSearch(""); }, 150);
  }

  function pick(option: ModelOption): void {
    setSelectedKey(option.key);
    onChange(option.path);
    setOpen(false);
    setSearch("");
  }

  function clear(): void {
    setSelectedKey("");
    onChange("");
    setSearch("");
  }

  function renderOptions(items: ModelOption[]) {
    return items.map((o) => (
      <li key={o.key}>
        <button
          type="button"
          className="dataset-select-option"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => pick(o)}
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
        {selectedKey && !open ? (
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
