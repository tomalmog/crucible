import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUp, ArrowDown, Plus, Search, Trash2, ChevronLeft, ChevronRight } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { listBenchmarks as listBenchmarksApi } from "../../api/studioApi";
import { AddBenchmarkModal } from "./AddBenchmarkModal";
import { BenchmarkDetailView } from "./BenchmarkDetailView";

// ── Types ────────────────────────────────────────────────────────────

export interface BenchmarkEntry {
  name: string;
  displayName: string;
  type: string;
  entries: number;
  description: string;
  lastRun: string | null;
  bestScore: number | null;
  localCompatible: boolean;
}

// ── Component ────────────────────────────────────────────────────────

type TypeFilter = "all" | "lm-eval" | "custom";

export function EvalTasksPage() {
  const { dataRoot } = useCrucible();
  const deleteCmd = useCrucibleCommand();
  const [sorting, setSorting] = useState<SortingState>([]);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const [showAdd, setShowAdd] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkEntry[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<BenchmarkEntry | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Poll for entry count updates after adding a benchmark
  const startPolling = useCallback(() => {
    stopPolling();
    let ticks = 0;
    pollRef.current = setInterval(async () => {
      ticks += 1;
      if (ticks > 15) { stopPolling(); return; } // stop after 30s
      try {
        const items = await listBenchmarksApi(dataRoot);
        const hasZero = items.some((b) => b.type === "lm-eval" && b.entries === 0);
        setBenchmarks(items.map((b) => ({
          name: b.name, displayName: b.displayName, type: b.type,
          entries: b.entries, description: b.description,
          lastRun: null, bestScore: null, localCompatible: b.localCompatible,
        })));
        if (!hasZero) stopPolling();
      } catch { /* ignore */ }
    }, 2000);
  }, [dataRoot, stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  const loadBenchmarks = useCallback(async () => {
    if (!dataRoot) return;
    try {
      const items = await listBenchmarksApi(dataRoot);
      setBenchmarks(
        items.map((b) => ({
          name: b.name,
          displayName: b.displayName,
          type: b.type,
          entries: b.entries,
          description: b.description,
          lastRun: null,
          bestScore: null,
          localCompatible: b.localCompatible,
        })),
      );
    } catch {
      setBenchmarks([]);
    }
  }, [dataRoot]);

  useEffect(() => {
    loadBenchmarks().catch(console.error);
  }, [loadBenchmarks]);

  const data = useMemo(() => {
    let rows = benchmarks;
    if (typeFilter !== "all") {
      rows = rows.filter((r) => r.type === typeFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((r) => r.displayName.toLowerCase().includes(q) || r.name.includes(q));
    }
    return rows;
  }, [search, typeFilter, benchmarks]);

  async function handleDelete() {
    if (!pendingDelete || !dataRoot) return;
    await deleteCmd.run(dataRoot, ["benchmark-registry", "delete", "--name", pendingDelete]);
    setPendingDelete(null);
    if (selectedBenchmark?.name === pendingDelete) setSelectedBenchmark(null);
    await loadBenchmarks();
  }

  const columns = useMemo(() => {
    const c = createColumnHelper<BenchmarkEntry>();
    return [
      c.accessor("displayName", {
        header: "Name",
        cell: (info) => <span className="registry-table-name">{info.getValue()}</span>,
      }),
      c.accessor("type", {
        header: "Type",
        cell: (info) => <span className="text-muted">{info.getValue()}</span>,
      }),
      c.accessor("localCompatible", {
        header: "Local",
        cell: (info) => info.getValue()
          ? <span style={{ color: "var(--success)", fontSize: "0.75rem" }}>supported</span>
          : <span className="text-muted" style={{ fontSize: "0.75rem" }} title="Uses text generation — crashes on macOS. Run on a remote cluster.">remote only</span>,
      }),
      c.accessor("entries", {
        header: "Entries",
        cell: (info) => {
          const v = info.getValue();
          return v > 0 ? v.toLocaleString() : <span className="text-muted">&mdash;</span>;
        },
      }),
      c.accessor("lastRun", {
        header: "Last Run",
        cell: (info) => (
          <span className="text-muted">{info.getValue() ?? "Never"}</span>
        ),
      }),
      c.accessor("bestScore", {
        header: "Best Score",
        cell: (info) => {
          const v = info.getValue();
          return v !== null ? `${v.toFixed(1)}%` : <span className="text-muted">&mdash;</span>;
        },
      }),
      c.display({
        id: "actions",
        header: "",
        cell: (info) => (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            title="Delete benchmark"
            onClick={(e) => { e.stopPropagation(); setPendingDelete(info.row.original.name); }}
          >
            <Trash2 size={14} />
          </button>
        ),
      }),
    ];
  }, []);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 15 } },
  });

  // ── Detail view ──────────────────────────────────────────────────

  if (selectedBenchmark) {
    return (
      <BenchmarkDetailView
        benchmark={selectedBenchmark}
        onBack={() => setSelectedBenchmark(null)}
      />
    );
  }

  // ── List view ────────────────────────────────────────────────────

  return (
    <>
      <PageHeader title="Benchmarks" />

      <div className="registry-toolbar">
        <div className="registry-search">
          <Search size={14} />
          <input
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            placeholder="Search benchmarks..."
          />
        </div>
        <div className="registry-filters">
          {(["all", "lm-eval", "custom"] as const).map((f) => (
            <button
              key={f}
              className={`registry-filter-chip ${typeFilter === f ? "registry-filter-chip--active" : ""}`}
              onClick={() => setTypeFilter(f)}
            >
              {f === "all" ? "All" : f === "lm-eval" ? "lm-eval" : "Custom"}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowAdd(true)}>
          <Plus size={14} /> New Benchmark
        </button>
      </div>

      <div className="registry-table-wrap">
        <table className="registry-table">
          <thead>
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={header.column.getCanSort() ? "sortable" : ""}
                  >
                    <div className="registry-th-content">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {{
                        asc: <ArrowUp size={12} />,
                        desc: <ArrowDown size={12} />,
                      }[header.column.getIsSorted() as string] ?? null}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => setSelectedBenchmark(row.original)}
                style={{ cursor: "pointer" }}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {table.getPageCount() > 1 && (
        <div className="dataset-pagination">
          <span className="text-muted text-sm">
            Showing {table.getState().pagination.pageIndex * 15 + 1}–
            {Math.min((table.getState().pagination.pageIndex + 1) * 15, data.length)} of {data.length}
          </span>
          <div className="pagination-controls">
            <button
              className="btn btn-ghost btn-sm btn-icon"
              disabled={!table.getCanPreviousPage()}
              onClick={() => table.previousPage()}
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-sm">
              Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
            </span>
            <button
              className="btn btn-ghost btn-sm btn-icon"
              disabled={!table.getCanNextPage()}
              onClick={() => table.nextPage()}
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}

      {showAdd && (
        <AddBenchmarkModal
          onAdded={() => { setShowAdd(false); loadBenchmarks().catch(console.error); startPolling(); }}
          onClose={() => setShowAdd(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Benchmark"
          itemName={pendingDelete}
          description="This will remove this benchmark from your registry."
          isDeleting={deleteCmd.isRunning}
          onConfirm={() => handleDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
