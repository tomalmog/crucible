import { useState, useEffect, useRef } from "react";
import { ArrowDownToLine, Heart, Download, Check, Loader, ChevronLeft, ChevronRight, SlidersHorizontal } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { PathInput } from "../../components/shared/PathInput";
import { FormField } from "../../components/shared/FormField";
import { formatCount, repoAuthor, formatDate } from "./hubUtils";
import { HubModelDetail } from "./HubModelDetail";

interface ModelResult {
  repo_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  task: string;
  last_modified: string;
}

type DownloadStatus = "idle" | "downloading" | "done" | "error";
const PER_PAGE = 12;

const TASK_OPTIONS = [
  "", "text-generation", "text-to-image", "text-classification",
  "fill-mask", "question-answering", "summarization", "translation",
  "conversational", "feature-extraction", "image-classification",
];

const LIBRARY_OPTIONS = [
  "", "transformers", "gguf", "safetensors", "pytorch", "onnx", "diffusers",
];

const SORT_OPTIONS = [
  { value: "downloads", label: "Downloads" },
  { value: "likes", label: "Likes" },
  { value: "created", label: "Newest" },
];

export function HubModelSearch() {
  const { dataRoot } = useForge();
  const searchCmd = useForgeCommand();
  const downloadCmd = useForgeCommand();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ModelResult[]>([]);
  const [page, setPage] = useState(0);
  const [targetDir, setTargetDir] = useState("./models");
  const [downloadStates, setDownloadStates] = useState<Record<string, DownloadStatus>>({});
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterTask, setFilterTask] = useState("");
  const [filterLibrary, setFilterLibrary] = useState("");
  const [filterSort, setFilterSort] = useState("downloads");
  const didLoad = useRef(false);

  const totalPages = Math.ceil(results.length / PER_PAGE);
  const pageResults = results.slice(page * PER_PAGE, (page + 1) * PER_PAGE);
  const hasFilters = filterTask || filterLibrary || filterSort !== "downloads";

  async function runSearch(searchQuery: string) {
    if (!dataRoot) return;
    const args = ["hub", "search-models", searchQuery, "--limit", "40", "--json"];
    if (filterTask) args.push("--filter", filterTask);
    if (filterLibrary) args.push("--library", filterLibrary);
    if (filterSort !== "downloads") args.push("--sort", filterSort);
    const status = await searchCmd.run(dataRoot, args);
    if (status.status === "completed" && status.stdout) {
      setResults(JSON.parse(status.stdout));
      setDownloadStates({});
      setPage(0);
    }
  }

  useEffect(() => {
    if (dataRoot && !didLoad.current) {
      didLoad.current = true;
      runSearch("llama").catch(console.error);
    }
  }, [dataRoot]);

  async function downloadModel(repoId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!dataRoot) return;
    setDownloadStates((s) => ({ ...s, [repoId]: "downloading" }));
    try {
      const status = await downloadCmd.run(dataRoot, [
        "hub", "download-model", repoId, "--target-dir", targetDir,
      ]);
      setDownloadStates((s) => ({
        ...s,
        [repoId]: status.status === "completed" ? "done" : "error",
      }));
    } catch {
      setDownloadStates((s) => ({ ...s, [repoId]: "error" }));
    }
  }

  if (selectedRepo) {
    return (
      <HubModelDetail
        repoId={selectedRepo}
        targetDir={targetDir}
        onBack={() => setSelectedRepo(null)}
      />
    );
  }

  return (
    <div className="hub-search-layout">
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
        <div className="hub-search-actions">
          <button
            className={`btn btn-sm ${showFilters || hasFilters ? "btn-primary" : ""}`}
            onClick={() => setShowFilters((v) => !v)}
            title="Filters"
          >
            <SlidersHorizontal size={13} />
          </button>
          <button
            className="btn btn-primary"
            onClick={() => runSearch(query).catch(console.error)}
            disabled={searchCmd.isRunning || !query.trim()}
          >
            {searchCmd.isRunning ? "Searching..." : "Search"}
          </button>
        </div>
      </div>

      {showFilters && (
        <div className="hub-filter-row">
          <label>
            <span>Task</span>
            <select value={filterTask} onChange={(e) => setFilterTask(e.target.value)}>
              <option value="">Any task</option>
              {TASK_OPTIONS.filter(Boolean).map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Library</span>
            <select value={filterLibrary} onChange={(e) => setFilterLibrary(e.target.value)}>
              <option value="">Any library</option>
              {LIBRARY_OPTIONS.filter(Boolean).map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Sort by</span>
            <select value={filterSort} onChange={(e) => setFilterSort(e.target.value)}>
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          {hasFilters && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => { setFilterTask(""); setFilterLibrary(""); setFilterSort("downloads"); }}
            >
              Clear
            </button>
          )}
        </div>
      )}

      <div className="hub-download-target">
        <FormField label="Download To">
          <PathInput value={targetDir} onChange={setTargetDir} placeholder="./models" kind="folder" />
        </FormField>
      </div>

      {searchCmd.error && <p className="error-text">{searchCmd.error}</p>}

      {searchCmd.isRunning && results.length === 0 && (
        <p className="text-tertiary text-xs">Loading models...</p>
      )}

      {pageResults.length > 0 && (
        <>
          <div className="hub-grid">
            {pageResults.map((r) => {
              const dlState = downloadStates[r.repo_id] ?? "idle";
              const author = repoAuthor(r.repo_id, r.author);
              return (
                <div className="hub-card" key={r.repo_id} onClick={() => setSelectedRepo(r.repo_id)}>
                  <div className="hub-card-header">
                    <div>
                      <div className="hub-card-repo">{r.repo_id}</div>
                      {author && <div className="hub-card-author">{author}</div>}
                    </div>
                    {r.task && <span className="hub-card-task">{r.task}</span>}
                  </div>

                  <div className="hub-card-bottom">
                    <div className="hub-card-stats">
                      <span className="hub-card-stat">
                        <ArrowDownToLine size={12} />
                        {formatCount(r.downloads)}
                      </span>
                      <span className="hub-card-stat">
                        <Heart size={12} />
                        {formatCount(r.likes)}
                      </span>
                      {r.last_modified && (
                        <span className="hub-card-stat hub-card-date">
                          {formatDate(r.last_modified)}
                        </span>
                      )}
                      <span className="hub-card-hint">View details</span>
                    </div>
                    <button
                      className={`btn btn-sm ${dlState === "done" ? "btn-success" : dlState === "error" ? "btn-error" : ""}`}
                      onClick={(e) => downloadModel(r.repo_id, e).catch(console.error)}
                      disabled={dlState === "downloading"}
                    >
                      {dlState === "downloading" && <><Loader size={12} className="spin" /> Downloading...</>}
                      {dlState === "done" && <><Check size={12} /> Downloaded</>}
                      {dlState === "error" && <><Download size={12} /> Retry</>}
                      {dlState === "idle" && <><Download size={12} /> Download</>}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {totalPages > 1 && (
            <div className="hub-pagination">
              <button className="btn btn-sm" onClick={() => setPage((p) => p - 1)} disabled={page === 0}>
                <ChevronLeft size={14} /> Prev
              </button>
              <span className="hub-pagination-info">
                {page + 1} / {totalPages}
              </span>
              <button className="btn btn-sm" onClick={() => setPage((p) => p + 1)} disabled={page >= totalPages - 1}>
                Next <ChevronRight size={14} />
              </button>
            </div>
          )}
        </>
      )}

      {!searchCmd.isRunning && !searchCmd.error && results.length === 0 && didLoad.current && (
        <p className="text-tertiary text-xs">No models found.</p>
      )}
    </div>
  );
}
