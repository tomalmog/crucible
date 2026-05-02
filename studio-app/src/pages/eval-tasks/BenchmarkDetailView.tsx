import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
} from "@tanstack/react-table";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { MetricCard } from "../../components/shared/MetricCard";
import { EmptyState } from "../../components/shared/EmptyState";
import type { BenchmarkEntry } from "./EvalTasksPage";

const PAGE_SIZE = 25;

interface SampleRow {
  index: number;
  prompt: string;
  response: string;
}

const col = createColumnHelper<SampleRow>();

function truncate(text: string, max = 120): string {
  return text.length > max ? text.slice(0, max) + "\u2026" : text;
}

// ── HuggingFace Dataset Viewer API ──────────────────────────────────────
// Fetches paginated rows directly from HF servers — no download, no Python
// subprocess. ~200ms per page vs ~3-10s with load_dataset.

interface HfViewerConfig {
  dataset: string;
  config: string;
  split: string;
  promptCol: string;
  responseCol: string;
}

const HF_VIEWER_DATASETS: Record<string, HfViewerConfig> = {
  mmlu:       { dataset: "cais/mmlu", config: "all", split: "test", promptCol: "question", responseCol: "choices" },
  hellaswag:  { dataset: "Rowan/hellaswag", config: "default", split: "validation", promptCol: "ctx", responseCol: "endings" },
  arc:        { dataset: "allenai/ai2_arc", config: "ARC-Challenge", split: "test", promptCol: "question", responseCol: "choices" },
  arc_easy:   { dataset: "allenai/ai2_arc", config: "ARC-Easy", split: "test", promptCol: "question", responseCol: "choices" },
  winogrande: { dataset: "allenai/winogrande", config: "winogrande_xl", split: "validation", promptCol: "sentence", responseCol: "option1" },
  truthfulqa: { dataset: "truthfulqa/truthful_qa", config: "multiple_choice", split: "validation", promptCol: "question", responseCol: "mc1_targets" },
  gsm8k:      { dataset: "openai/gsm8k", config: "main", split: "test", promptCol: "question", responseCol: "answer" },
  math:       { dataset: "EleutherAI/hendrycks_math", config: "algebra", split: "test", promptCol: "problem", responseCol: "solution" },
  bbh:        { dataset: "lukaemon/bbh", config: "boolean_expressions", split: "test", promptCol: "input", responseCol: "target" },
  humaneval:  { dataset: "openai/openai_humaneval", config: "openai_humaneval", split: "test", promptCol: "prompt", responseCol: "canonical_solution" },
  mbpp:       { dataset: "google-research-datasets/mbpp", config: "full", split: "test", promptCol: "text", responseCol: "code" },
  boolq:      { dataset: "google/boolq", config: "default", split: "validation", promptCol: "question", responseCol: "answer" },
  openbookqa: { dataset: "allenai/openbookqa", config: "main", split: "test", promptCol: "question_stem", responseCol: "choices" },
};

async function fetchFromHfViewer(
  cfg: HfViewerConfig, offset: number, limit: number,
): Promise<{ rows: { prompt: string; response: string }[]; total: number } | null> {
  const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(cfg.dataset)}&config=${encodeURIComponent(cfg.config)}&split=${encodeURIComponent(cfg.split)}&offset=${offset}&length=${limit}`;
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.error || !data.rows) return null;
    const rows = data.rows.map((r: { row: Record<string, unknown> }) => {
      const raw = r.row;
      const p = raw[cfg.promptCol] ?? "";
      const resp = raw[cfg.responseCol] ?? "";
      return {
        prompt: typeof p === "string" ? p : JSON.stringify(p),
        response: typeof resp === "string" ? resp : JSON.stringify(resp),
      };
    });
    return { rows, total: data.num_rows_total ?? rows.length };
  } catch {
    return null;
  }
}

// ── Component ───────────────────────────────────────────────────────────

interface BenchmarkDetailViewProps {
  benchmark: BenchmarkEntry;
  onBack: () => void;
}

export function BenchmarkDetailView({ benchmark, onBack }: BenchmarkDetailViewProps) {
  const { dataRoot } = useCrucible();
  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const isCustom = benchmark.type === "custom";
  const dataPath = isCustom ? `${dataRoot}/benchmarks/${benchmark.name}/data.jsonl` : null;
  const hfCfg = !isCustom ? HF_VIEWER_DATASETS[benchmark.name] : undefined;

  const fetchSamples = useCallback(async (pageNum: number) => {
    setLoading(true);
    const offset = pageNum * PAGE_SIZE;
    try {
      // Fast path: HF Dataset Viewer API (no subprocess, ~200ms)
      if (hfCfg) {
        const result = await fetchFromHfViewer(hfCfg, offset, PAGE_SIZE);
        if (result && result.rows.length > 0) {
          setSamples(result.rows.map((r, i) => ({ index: offset + i, prompt: r.prompt, response: r.response })));
          setTotalCount(result.total);
          return;
        }
        // Fall through to CLI subprocess if API failed
      }

      // Slow fallback: CLI subprocess (custom benchmarks or API unavailable)
      const { startCrucibleCommand, getCrucibleCommandStatus } = await import("../../api/studioApi");
      const args = ["benchmark-registry", "samples", "--name", benchmark.name, "--offset", String(offset), "--limit", String(PAGE_SIZE)];
      if (!isCustom) args.push("--builtin");
      const task = await startCrucibleCommand(dataRoot, args);
      let status = await getCrucibleCommandStatus(task.task_id);
      while (status.status === "running") {
        await new Promise((r) => setTimeout(r, 300));
        status = await getCrucibleCommandStatus(task.task_id);
      }
      if (status.status === "completed" && status.stdout.trim()) {
        const parsed = JSON.parse(status.stdout);
        const rows = parsed.rows ?? parsed;
        if (benchmark.entries > 0) setTotalCount(benchmark.entries);
        else if (parsed.total !== undefined) setTotalCount(parsed.total);
        setSamples(rows.map((r: { prompt: string; response: string }, i: number) => ({
          index: offset + i,
          prompt: r.prompt,
          response: r.response,
        })));
      }
    } catch {
      setSamples([]);
    } finally {
      setLoading(false);
    }
  }, [benchmark.name, benchmark.entries, dataRoot, isCustom, hfCfg]);

  useEffect(() => {
    setPage(0);
    setSamples([]);
    setTotalCount(benchmark.entries);
    fetchSamples(0).catch(console.error);
  }, [benchmark.name, fetchSamples, benchmark.entries]);

  useEffect(() => {
    if (page > 0) {
      setSamples([]);
      fetchSamples(page).catch(console.error);
    }
  }, [page, fetchSamples]);

  const columns = useMemo(() => [
    col.accessor("prompt", {
      header: "Prompt",
      cell: (info) => truncate(info.getValue(), 100),
      size: 50,
    }),
    col.accessor("response", {
      header: "Response",
      cell: (info) => truncate(info.getValue(), 100),
      size: 50,
    }),
  ], []);

  const table = useReactTable({
    data: samples,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));
  const start = page * PAGE_SIZE + 1;
  const end = Math.min((page + 1) * PAGE_SIZE, totalCount);

  return (
    <>
      <div style={{ marginBottom: 16 }}>
        <button className="btn btn-ghost btn-sm" onClick={onBack} style={{ marginLeft: -8 }}>
          <ChevronLeft size={14} /> Back
        </button>
      </div>

      <div className="stack-lg">
        <div className="stats-grid">
          <MetricCard label="Name" value={benchmark.displayName} />
          <MetricCard label="Type" value={benchmark.type} />
          <MetricCard label="Entries" value={benchmark.entries.toLocaleString()} />
          <MetricCard label="Local" value={benchmark.localCompatible ? "Supported" : "Remote only"} />
          {benchmark.bestScore !== null && (
            <MetricCard label="Best Score" value={`${benchmark.bestScore.toFixed(1)}%`} />
          )}
        </div>

        {!benchmark.localCompatible && (
          <div className="info-banner">
            This benchmark uses text generation which crashes on macOS due to a Python multiprocessing bug.
            Run it on a remote Linux cluster instead.
          </div>
        )}

        <div className="panel">
          <h4 className="panel-title">Description</h4>
          <p className="text-muted" style={{ margin: 0, fontSize: "0.8125rem" }}>{benchmark.description}</p>
        </div>

        {isCustom && dataPath && (
          <div className="panel">
            <h4 className="panel-title">Path</h4>
            <code className="text-mono text-sm text-muted" style={{ wordBreak: "break-all" }}>{dataPath}</code>
          </div>
        )}

        {/* Samples table */}
        <>
            {loading && samples.length === 0 ? (
              <EmptyState title="Loading..." description="Fetching samples." />
            ) : samples.length === 0 ? (
              <EmptyState title="No samples" description="This benchmark has no entries." />
            ) : (
              <>
                <div className="registry-table-wrap">
                  <table className="registry-table equal-cols">
                    <thead>
                      {table.getHeaderGroups().map((hg) => (
                        <tr key={hg.id}>
                          {hg.headers.map((header) => (
                            <th key={header.id}>
                              {flexRender(header.column.columnDef.header, header.getContext())}
                            </th>
                          ))}
                        </tr>
                      ))}
                    </thead>
                    <tbody>
                      {table.getRowModel().rows.map((row) => {
                        const isExpanded = expandedIdx === row.original.index;
                        return (
                          <>
                            <tr
                              key={row.id}
                              onClick={() => setExpandedIdx(isExpanded ? null : row.original.index)}
                              style={{ cursor: "pointer" }}
                              className={isExpanded ? "row-expanded" : undefined}
                            >
                              {row.getVisibleCells().map((cell) => (
                                <td key={cell.id}>
                                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                </td>
                              ))}
                            </tr>
                            {isExpanded && (
                              <tr key={`${row.id}-detail`} className="row-detail">
                                <td colSpan={2}>
                                  <div className="sample-detail">
                                    <div className="sample-detail-field">
                                      <strong>Prompt</strong>
                                      <pre>{row.original.prompt}</pre>
                                    </div>
                                    <div className="sample-detail-field">
                                      <strong>Response</strong>
                                      <pre>{row.original.response}</pre>
                                    </div>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="dataset-pagination">
                  <span className="text-muted text-sm">
                    Showing {start}–{end} of {totalCount.toLocaleString()} entries
                  </span>
                  <div className="pagination-controls">
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      disabled={page === 0}
                      onClick={() => setPage(page - 1)}
                    >
                      <ChevronLeft size={14} />
                    </button>
                    <span className="text-sm">
                      Page {page + 1} of {totalPages}
                    </span>
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      disabled={page >= totalPages - 1}
                      onClick={() => setPage(page + 1)}
                    >
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              </>
            )}
          </>
      </div>
    </>
  );
}
