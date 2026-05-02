import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";

interface ModelMultiSelectProps {
  selected: Set<string>;
  onChange: (selected: Set<string>) => void;
  placeholder?: string;
}

interface ModelOption {
  label: string;
  key: string;
  path: string;
  section: "local" | "remote";
}

export function ModelMultiSelect({
  selected,
  onChange,
  placeholder = "Select models",
}: ModelMultiSelectProps) {
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

  const allOptions = useMemo(() => {
    const opts: ModelOption[] = [];
    const knownHosts = new Set(hostToCluster.keys());
    for (const m of models) {
      if (m.hasLocal && m.modelPath) {
        opts.push({ label: m.modelName, key: m.modelPath, path: m.modelPath, section: "local" });
      }
      if (m.hasRemote && m.remotePath && knownHosts.has(m.remoteHost)) {
        const clusterLabel = hostToCluster.get(m.remoteHost) || m.remoteHost;
        opts.push({ label: `${m.modelName} — ${clusterLabel}`, key: m.remotePath, path: m.remotePath, section: "remote" });
      }
    }
    return opts;
  }, [models, hostToCluster]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return q ? allOptions.filter((o) => o.label.toLowerCase().includes(q)) : allOptions;
  }, [allOptions, search]);

  const localOptions = filtered.filter((o) => o.section === "local");
  const remoteGroups = useMemo(() => {
    const groups = new Map<string, ModelOption[]>();
    for (const o of filtered) {
      if (o.section !== "remote") continue;
      const host = o.label.split(" — ").pop() || "";
      const list = groups.get(host) || [];
      list.push(o);
      groups.set(host, list);
    }
    return groups;
  }, [filtered]);

  const hasResults = localOptions.length > 0 || remoteGroups.size > 0;

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setSearch("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => { setOpen(false); setSearch(""); }, 150);
  }

  function toggle(path: string): void {
    const next = new Set(selected);
    if (next.has(path)) next.delete(path); else next.add(path);
    onChange(next);
  }

  const displayText = selected.size === 0
    ? ""
    : selected.size === 1
      ? (allOptions.find((o) => selected.has(o.path))?.label ?? `${selected.size} model`)
      : `${selected.size} models selected`;

  function renderOptions(items: ModelOption[]) {
    return items.map((o) => (
      <li key={o.key}>
        <button
          type="button"
          className="dataset-select-option"
          style={{ display: "flex", alignItems: "center", gap: 8 }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => toggle(o.path)}
        >
          <input
            type="checkbox"
            checked={selected.has(o.path)}
            readOnly
            style={{ width: "auto", margin: 0, padding: 0, border: "none", boxShadow: "none", flexShrink: 0 }}
          />
          {o.label}
        </button>
      </li>
    ));
  }

  return (
    <div className="dataset-select" onFocus={handleFocus} onBlur={handleBlur}>
      <div className="dataset-select-input-wrap">
        <input
          value={open ? search : displayText}
          onChange={(e) => setSearch(e.currentTarget.value)}
          onClick={() => { if (!open) { setSearch(""); setOpen(true); } }}
          placeholder={displayText || placeholder}
          readOnly={!open}
        />
        <ChevronDown size={14} className="dataset-select-chevron" />
      </div>
      {open && (
        <ul className="dataset-select-dropdown" style={{ maxHeight: 280 }}>
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
