import { useState, useEffect, useRef } from "react";
import { Download, Heart, ArrowDownToLine } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";

interface ModelResult {
  repo_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  task: string;
}

const DISPLAY_TAGS = ["transformers", "safetensors", "pytorch", "gguf", "onnx", "text-generation", "text-to-image", "conversational", "fill-mask", "question-answering", "summarization", "translation"];

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function visibleTags(tags: string[]): string[] {
  return tags.filter((t) => DISPLAY_TAGS.includes(t)).slice(0, 5);
}

function repoAuthor(result: ModelResult): string {
  if (result.author) return result.author;
  const slash = result.repo_id.indexOf("/");
  return slash > 0 ? result.repo_id.slice(0, slash) : "";
}

export function HubModelSearch() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ModelResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const didLoad = useRef(false);

  async function runSearch(searchQuery: string) {
    if (!dataRoot) return;
    setSearching(true);
    setError(null);
    try {
      const status = await command.run(dataRoot, [
        "hub", "search-models", searchQuery, "--limit", "12", "--json",
      ]);
      if (status.status === "completed" && status.stdout) {
        const parsed: ModelResult[] = JSON.parse(status.stdout);
        setResults(parsed);
      } else if (status.status === "failed") {
        setError(status.stderr || "Search failed.");
      }
    } catch {
      setError("Failed to reach forge CLI.");
    }
    setSearching(false);
  }

  useEffect(() => {
    if (dataRoot && !didLoad.current) {
      didLoad.current = true;
      runSearch("llama").catch(console.error);
    }
  }, [dataRoot]);

  async function downloadModel(repoId: string) {
    if (!dataRoot) return;
    await command.run(dataRoot, ["hub", "download-model", repoId, "--target-dir", "./models"]);
  }

  return (
    <div className="stack">
      <div className="hub-search-row">
        <label>
          <span>Search Models</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
            placeholder="llama, mistral, phi, bert..."
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
        <p className="text-tertiary" style={{ fontSize: "0.75rem" }}>Loading models...</p>
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
                {r.task && <span className="badge badge-accent">{r.task}</span>}
              </div>

              <div className="hub-card-stats">
                <span className="hub-card-stat">
                  <ArrowDownToLine />
                  {formatCount(r.downloads)}
                </span>
                <span className="hub-card-stat">
                  <Heart />
                  {formatCount(r.likes)}
                </span>
              </div>

              {visibleTags(r.tags).length > 0 && (
                <div className="hub-card-tags">
                  {visibleTags(r.tags).map((t) => (
                    <span className="hub-tag" key={t}>{t}</span>
                  ))}
                </div>
              )}

              <div className="hub-card-footer">
                <button className="btn btn-sm" onClick={() => downloadModel(r.repo_id).catch(console.error)}>
                  <Download /> Download
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {!searching && !error && results.length === 0 && didLoad.current && (
        <p className="text-tertiary" style={{ fontSize: "0.75rem" }}>No models found.</p>
      )}
    </div>
  );
}
