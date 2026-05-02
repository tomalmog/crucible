import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { listBenchmarks } from "../../api/studioApi";

interface BenchmarkMultiSelectProps {
  selected: Set<string>;
  onChange: (selected: Set<string>) => void;
  placeholder?: string;
}

interface BenchmarkOption {
  name: string;
  displayName: string;
  localCompatible: boolean;
}

export function BenchmarkMultiSelect({
  selected,
  onChange,
  placeholder = "Select benchmarks",
}: BenchmarkMultiSelectProps) {
  const { dataRoot } = useCrucible();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [benchmarks, setBenchmarks] = useState<BenchmarkOption[]>([]);
  const blurTimeout = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    if (!dataRoot) return;
    listBenchmarks(dataRoot)
      .then((items) => setBenchmarks(items.map((b) => ({
        name: b.name,
        displayName: b.displayName,
        localCompatible: b.localCompatible,
      }))))
      .catch(() => setBenchmarks([]));
  }, [dataRoot]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return q ? benchmarks.filter((b) => b.displayName.toLowerCase().includes(q) || b.name.includes(q)) : benchmarks;
  }, [benchmarks, search]);

  const localOptions = filtered.filter((b) => b.localCompatible);
  const remoteOptions = filtered.filter((b) => !b.localCompatible);
  const hasResults = localOptions.length > 0 || remoteOptions.length > 0;

  function handleFocus(): void {
    clearTimeout(blurTimeout.current);
    setSearch("");
    setOpen(true);
  }

  function handleBlur(): void {
    blurTimeout.current = setTimeout(() => { setOpen(false); setSearch(""); }, 150);
  }

  function toggle(name: string): void {
    const next = new Set(selected);
    if (next.has(name)) next.delete(name); else next.add(name);
    onChange(next);
  }

  const displayText = selected.size === 0
    ? ""
    : selected.size === 1
      ? (benchmarks.find((b) => selected.has(b.name))?.displayName ?? `${selected.size} benchmark`)
      : `${selected.size} benchmarks selected`;

  function renderOptions(items: BenchmarkOption[]) {
    return items.map((b) => (
      <li key={b.name}>
        <button
          type="button"
          className="dataset-select-option"
          style={{ display: "flex", alignItems: "center", gap: 8 }}
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => toggle(b.name)}
        >
          <input
            type="checkbox"
            checked={selected.has(b.name)}
            readOnly
            style={{ width: "auto", margin: 0, padding: 0, border: "none", boxShadow: "none", flexShrink: 0 }}
          />
          {b.displayName}
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
          {remoteOptions.length > 0 && (
            <>
              <li className="dataset-select-header">Remote Only</li>
              {renderOptions(remoteOptions)}
            </>
          )}
          {!hasResults && (
            <li className="dataset-select-empty">No benchmarks found</li>
          )}
        </ul>
      )}
    </div>
  );
}
