import { useEffect, useState } from "react";
import { Plus, Search, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { FormField } from "../../components/shared/FormField";
import { PathInput } from "../../components/shared/PathInput";

interface AddBenchmarkModalProps {
  onAdded: () => void;
  onClose: () => void;
}

type Tab = "browse" | "custom";

function extractError(raw: string): string {
  const lines = raw.trim().split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.match(/^[\w.]*Error:\s/) || line.match(/^[\w.]*Exception:\s/)) {
      return line.replace(/^[\w.]*(?:Error|Exception):\s*/, "");
    }
  }
  return raw;
}

export function AddBenchmarkModal({ onAdded, onClose }: AddBenchmarkModalProps) {
  const { dataRoot } = useCrucible();
  const command = useCrucibleCommand();
  const searchCmd = useCrucibleCommand();
  const [tab, setTab] = useState<Tab>("browse");

  // Browse tab state
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ name: string }[]>([]);
  const [adding, setAdding] = useState<string | null>(null);

  // Custom tab state
  const [name, setName] = useState("");
  const [source, setSource] = useState("");

  const busy = command.isRunning;

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && !busy) onClose();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [busy, onClose]);

  async function handleSearch() {
    if (!query.trim() || !dataRoot) return;
    const result = await searchCmd.run(dataRoot, ["benchmark-registry", "search", query.trim(), "--limit", "20"]);
    if (result.status === "completed" && result.stdout.trim()) {
      try {
        setResults(JSON.parse(result.stdout));
      } catch {
        setResults([]);
      }
    }
  }

  async function handleAddLmEval(taskName: string) {
    if (!dataRoot) return;
    setAdding(taskName);
    const result = await command.run(dataRoot, ["benchmark-registry", "add", "--name", taskName]);
    if (result.status === "completed") {
      // Fire background job to resolve entry count (don't wait)
      const { startCrucibleCommand } = await import("../../api/studioApi");
      startCrucibleCommand(dataRoot, ["benchmark-registry", "resolve-count", "--name", taskName]).catch(() => {});
      onAdded();
    }
    setAdding(null);
  }

  async function handleCreateCustom(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !source.trim() || !dataRoot) return;
    const result = await command.run(dataRoot, [
      "benchmark-registry", "create", "--name", name.trim(), "--source", source.trim(),
    ]);
    if (result.status === "completed") {
      onAdded();
    }
  }

  return (
    <div className="modal-backdrop" onClick={!busy ? onClose : undefined}>
      <div className="confirm-modal" style={{ width: 520 }} onClick={(e) => e.stopPropagation()}>
        <div className="confirm-modal-header">
          <h3 className="confirm-modal-title">New Benchmark</h3>
          {!busy && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>

        {/* Tabs */}
        <div className="modal-tab-bar">
          <button
            className={`modal-tab ${tab === "browse" ? "modal-tab--active" : ""}`}
            onClick={() => setTab("browse")}
          >
            Browse lm-eval
          </button>
          <button
            className={`modal-tab ${tab === "custom" ? "modal-tab--active" : ""}`}
            onClick={() => setTab("custom")}
          >
            Custom JSONL
          </button>
        </div>

        {/* Browse tab */}
        {tab === "browse" && (
          <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div className="registry-search" style={{ maxWidth: "none" }}>
              <Search size={14} />
              <input
                value={query}
                onChange={(e) => setQuery(e.currentTarget.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleSearch().catch(console.error); }}
                placeholder="Search 14,000+ tasks..."
              />
            </div>
            <button
              className="btn btn-sm"
              onClick={() => handleSearch().catch(console.error)}
              disabled={searchCmd.isRunning || !query.trim()}
            >
              {searchCmd.isRunning ? "Searching..." : "Search"}
            </button>

            {results.length > 0 && (
              <div className="benchmark-search-results">
                {results.map((r) => (
                  <div key={r.name} className="benchmark-search-row">
                    <span className="text-sm">{r.name}</span>
                    <button
                      className="btn btn-ghost btn-sm"
                      onClick={() => handleAddLmEval(r.name).catch(console.error)}
                      disabled={adding === r.name}
                    >
                      {adding === r.name ? "Adding..." : <><Plus size={12} /> Add</>}
                    </button>
                  </div>
                ))}
              </div>
            )}

            {!busy && command.error && (
              <p className="error-text">{extractError(command.error)}</p>
            )}
          </div>
        )}

        {/* Custom tab */}
        {tab === "custom" && (
          <form onSubmit={(e) => handleCreateCustom(e).catch(console.error)}>
            <div className="confirm-modal-body" style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <FormField label="Name">
                <input
                  value={name}
                  onChange={(e) => setName(e.currentTarget.value)}
                  placeholder="my-benchmark"
                  disabled={busy}
                />
              </FormField>
              <FormField label="Data file" hint="JSONL with prompt and response fields">
                <PathInput
                  value={source}
                  onChange={setSource}
                  placeholder="/path/to/questions.jsonl"
                  kind="file"
                  filters={[{ name: "JSONL files", extensions: ["jsonl"] }]}
                  disabled={busy}
                />
              </FormField>

              {!busy && command.error && (
                <p className="error-text">{extractError(command.error)}</p>
              )}
            </div>
            <div className="confirm-modal-footer">
              <button type="button" className="btn btn-sm" onClick={onClose}>Cancel</button>
              <button
                type="submit"
                className="btn btn-sm btn-primary"
                disabled={busy || !name.trim() || !source.trim()}
              >
                {busy ? "Creating..." : "Create"}
              </button>
            </div>
          </form>
        )}

        {/* Footer for browse tab */}
        {tab === "browse" && (
          <div className="confirm-modal-footer">
            <button className="btn btn-sm" onClick={onClose}>Close</button>
          </div>
        )}
      </div>
    </div>
  );
}
