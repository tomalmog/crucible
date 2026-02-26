import { useState, useEffect, useRef } from "react";
import { Download, ArrowDownToLine } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";

interface DatasetResult {
  repo_id: string;
  author: string;
  downloads: number;
  tags: string[];
}

const DISPLAY_TAGS = ["format:parquet", "modality:text", "modality:tabular", "modality:image", "library:datasets", "license:mit", "license:apache-2.0", "license:cc-by-4.0"];

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function visibleTags(tags: string[]): string[] {
  return tags.filter((t) => DISPLAY_TAGS.includes(t)).slice(0, 4);
}

function sizeTag(tags: string[]): string | null {
  const match = tags.find((t) => t.startsWith("size_categories:"));
  return match ? match.replace("size_categories:", "") : null;
}

function repoAuthor(result: DatasetResult): string {
  if (result.author) return result.author;
  const slash = result.repo_id.indexOf("/");
  return slash > 0 ? result.repo_id.slice(0, slash) : "";
}

export function HubDatasetSearch() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<DatasetResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const didLoad = useRef(false);

  async function runSearch(searchQuery: string) {
    if (!dataRoot) return;
    setSearching(true);
    setError(null);
    try {
      const status = await command.run(dataRoot, [
        "hub", "search-datasets", searchQuery, "--limit", "12", "--json",
      ]);
      if (status.status === "completed" && status.stdout) {
        const parsed: DatasetResult[] = JSON.parse(status.stdout);
        setResults(parsed);
      } else if (status.status === "failed") {
        setError(status.stderr || "Search failed.");
      }
    } catch {
      setError("Failed to reach forge CLI.");
    }
    setSearching(false);
  }

  // Run default search on first mount once dataRoot is available
  useEffect(() => {
    if (dataRoot && !didLoad.current) {
      didLoad.current = true;
      runSearch("instruction").catch(console.error);
    }
  }, [dataRoot]);

  async function downloadDataset(repoId: string) {
    if (!dataRoot) return;
    await command.run(dataRoot, ["hub", "download-dataset", repoId]);
  }

  return (
    <div className="stack">
      <div className="hub-search-row">
        <label>
          <span>Search Datasets</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
            placeholder="instruction-tuning, code, chat..."
            onKeyDown={(e) => e.key === "Enter" && query.trim() && runSearch(query).catch(console.error)}
          />
        </label>
        <button
          className="btn btn-primary"
          onClick={() => runSearch(query).catch(console.error)}
          disabled={searching || !query.trim()}
        >
          {searching ? "Searching..." : "Search"}
        </button>
      </div>

      {error && <p className="error-text">{error}</p>}

      {searching && results.length === 0 && (
        <p className="text-tertiary text-xs">Loading datasets...</p>
      )}

      {results.length > 0 && (
        <div className="hub-grid">
          {results.map((r) => (
            <div className="hub-card" key={r.repo_id}>
              <div className="hub-card-header">
                <div>
                  <div className="hub-card-repo">{r.repo_id}</div>
                  {repoAuthor(r) && <div className="hub-card-author">{repoAuthor(r)}</div>}
                </div>
                {sizeTag(r.tags) && <span className="badge badge-accent">{sizeTag(r.tags)}</span>}
              </div>

              <div className="hub-card-stats">
                <span className="hub-card-stat">
                  <ArrowDownToLine />
                  {formatCount(r.downloads)}
                </span>
              </div>

              {visibleTags(r.tags).length > 0 && (
                <div className="hub-card-tags">
                  {visibleTags(r.tags).map((t) => (
                    <span className="hub-tag" key={t}>{t.replace(/^[^:]+:/, "")}</span>
                  ))}
                </div>
              )}

              <div className="hub-card-footer">
                <button className="btn btn-sm" onClick={() => downloadDataset(r.repo_id).catch(console.error)}>
                  <Download /> Download
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {!searching && !error && results.length === 0 && didLoad.current && (
        <p className="text-tertiary text-xs">No datasets found.</p>
      )}
    </div>
  );
}
