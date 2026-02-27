import { useState, useEffect, useRef } from "react";
import { ArrowDownToLine, Download, Check, Loader, ChevronLeft, ChevronRight, SlidersHorizontal } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { PathInput } from "../../components/shared/PathInput";
import { FormField } from "../../components/shared/FormField";
import { formatCount, repoAuthor, formatDate, sizeTag } from "./hubUtils";
import { HubDatasetDetail } from "./HubDatasetDetail";

interface DatasetResult {
  repo_id: string;
  author: string;
  downloads: number;
  tags: string[];
  last_modified: string;
}

type DownloadStatus = "idle" | "downloading" | "done" | "error";
const PER_PAGE = 12;

const TASK_OPTIONS = [
  "", "text-classification", "question-answering", "summarization",
  "translation", "text-generation", "conversational", "text2text-generation",
  "token-classification", "image-classification",
];

const SORT_OPTIONS = [
  { value: "downloads", label: "Downloads" },
  { value: "likes", label: "Likes" },
  { value: "createdAt", label: "Newest" },
];

export function HubDatasetSearch() {
  const { dataRoot } = useForge();
  const searchCmd = useForgeCommand();
  const downloadCmd = useForgeCommand();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<DatasetResult[]>([]);
  const [page, setPage] = useState(0);
  const [targetDir, setTargetDir] = useState("./datasets");
  const [downloadStates, setDownloadStates] = useState<Record<string, DownloadStatus>>({});
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filterTask, setFilterTask] = useState("");
  const [filterSort, setFilterSort] = useState("downloads");
  const didLoad = useRef(false);

  const totalPages = Math.ceil(results.length / PER_PAGE);
  const pageResults = results.slice(page * PER_PAGE, (page + 1) * PER_PAGE);
  const hasFilters = filterTask || filterSort !== "downloads";

  async function runSearch(searchQuery: string) {
    if (!dataRoot) return;
    const args = ["hub", "search-datasets", searchQuery, "--limit", "40", "--json"];
    if (filterTask) args.push("--filter", filterTask);
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
      runSearch("instruction").catch(console.error);
    }
  }, [dataRoot]);

  async function downloadDataset(repoId: string, e: React.MouseEvent) {
    e.stopPropagation();
    if (!dataRoot) return;
    setDownloadStates((s) => ({ ...s, [repoId]: "downloading" }));
    try {
      const status = await downloadCmd.run(dataRoot, [
        "hub", "download-dataset", repoId, "--target-dir", targetDir,
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
      <HubDatasetDetail
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
          <span>Search Datasets</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.currentTarget.value)}
            placeholder="instruction-tuning, code, chat..."
            onKeyDown={(e) => e.key === "Enter" && (query.trim() || hasFilters) && runSearch(query).catch(console.error)}
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
            disabled={searchCmd.isRunning || (!query.trim() && !hasFilters)}
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
              onClick={() => { setFilterTask(""); setFilterSort("downloads"); }}
            >
              Clear
            </button>
          )}
        </div>
      )}

      <div className="hub-download-target">
        <FormField label="Download To">
          <PathInput value={targetDir} onChange={setTargetDir} placeholder="./datasets" kind="folder" />
        </FormField>
      </div>

      {searchCmd.error && <p className="error-text">{searchCmd.error}</p>}

      {searchCmd.isRunning && results.length === 0 && (
        <p className="text-tertiary text-xs">Loading datasets...</p>
      )}

      {pageResults.length > 0 && (
        <>
          <div className="hub-grid">
            {pageResults.map((r) => {
              const dlState = downloadStates[r.repo_id] ?? "idle";
              const author = repoAuthor(r.repo_id, r.author);
              const size = sizeTag(r.tags);
              return (
                <div className="hub-card" key={r.repo_id} onClick={() => setSelectedRepo(r.repo_id)}>
                  <div className="hub-card-header">
                    <div>
                      <div className="hub-card-repo">{r.repo_id}</div>
                      {author && <div className="hub-card-author">{author}</div>}
                    </div>
                    {size && <span className="hub-card-task">{size}</span>}
                  </div>

                  <div className="hub-card-bottom">
                    <div className="hub-card-stats">
                      <span className="hub-card-stat">
                        <ArrowDownToLine size={12} />
                        {formatCount(r.downloads)}
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
                      onClick={(e) => downloadDataset(r.repo_id, e).catch(console.error)}
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
        <p className="text-tertiary text-xs">No datasets found.</p>
      )}
    </div>
  );
}
