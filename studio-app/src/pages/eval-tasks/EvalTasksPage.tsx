import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { ArrowUp, ArrowDown, Plus, Search, Trash2 } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { CreateBenchmarkModal } from "./CreateBenchmarkModal";

// ── Benchmark data ───────────────────────────────────────────────────

interface BenchmarkEntry {
  name: string;
  displayName: string;
  type: "built-in" | "custom";
  entries: number;
  lastRun: string | null;
  bestScore: number | null;
}

const BUILT_IN: BenchmarkEntry[] = [
  { name: "mmlu", displayName: "MMLU", type: "built-in", entries: 14042, lastRun: null, bestScore: null },
  { name: "hellaswag", displayName: "HellaSwag", type: "built-in", entries: 10042, lastRun: null, bestScore: null },
  { name: "arc", displayName: "ARC Challenge", type: "built-in", entries: 1172, lastRun: null, bestScore: null },
  { name: "arc_easy", displayName: "ARC Easy", type: "built-in", entries: 2376, lastRun: null, bestScore: null },
  { name: "winogrande", displayName: "WinoGrande", type: "built-in", entries: 1267, lastRun: null, bestScore: null },
  { name: "truthfulqa", displayName: "TruthfulQA", type: "built-in", entries: 817, lastRun: null, bestScore: null },
  { name: "gsm8k", displayName: "GSM8K", type: "built-in", entries: 1319, lastRun: null, bestScore: null },
  { name: "math", displayName: "MATH", type: "built-in", entries: 5000, lastRun: null, bestScore: null },
  { name: "bbh", displayName: "BBH", type: "built-in", entries: 6511, lastRun: null, bestScore: null },
  { name: "humaneval", displayName: "HumanEval", type: "built-in", entries: 164, lastRun: null, bestScore: null },
  { name: "mbpp", displayName: "MBPP", type: "built-in", entries: 500, lastRun: null, bestScore: null },
  { name: "boolq", displayName: "BoolQ", type: "built-in", entries: 3270, lastRun: null, bestScore: null },
  { name: "piqa", displayName: "PIQA", type: "built-in", entries: 1838, lastRun: null, bestScore: null },
  { name: "openbookqa", displayName: "OpenBookQA", type: "built-in", entries: 500, lastRun: null, bestScore: null },
];

// ── Component ────────────────────────────────────────────────────────

type TypeFilter = "all" | "built-in" | "custom";

export function EvalTasksPage() {
  const { dataRoot } = useCrucible();
  const listCmd = useCrucibleCommand();
  const deleteCmd = useCrucibleCommand();
  const [sorting, setSorting] = useState<SortingState>([]);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const [showCreate, setShowCreate] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [customBenchmarks, setCustomBenchmarks] = useState<BenchmarkEntry[]>([]);

  const loadCustom = useCallback(async () => {
    if (!dataRoot) return;
    const result = await listCmd.run(dataRoot, ["benchmark-registry", "list", "--json"]);
    if (result.status === "completed" && result.stdout.trim()) {
      try {
        const items = JSON.parse(result.stdout);
        setCustomBenchmarks(
          items.map((b: { name: string; entries: number; created_at: string }) => ({
            name: b.name,
            displayName: b.name,
            type: "custom" as const,
            entries: b.entries,
            lastRun: null,
            bestScore: null,
          })),
        );
      } catch {
        // stdout might not be JSON (e.g. "No custom benchmarks.")
        setCustomBenchmarks([]);
      }
    }
  }, [dataRoot]);

  useEffect(() => {
    loadCustom().catch(console.error);
  }, [loadCustom]);

  const data = useMemo(() => {
    let rows = [...BUILT_IN, ...customBenchmarks];
    if (typeFilter !== "all") {
      rows = rows.filter((r) => r.type === typeFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((r) => r.displayName.toLowerCase().includes(q) || r.name.includes(q));
    }
    return rows;
  }, [search, typeFilter, customBenchmarks]);

  async function handleDelete() {
    if (!pendingDelete || !dataRoot) return;
    await deleteCmd.run(dataRoot, ["benchmark-registry", "delete", "--name", pendingDelete]);
    setPendingDelete(null);
    await loadCustom();
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
      c.accessor("entries", {
        header: "Entries",
        cell: (info) => info.getValue().toLocaleString(),
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
        cell: (info) => {
          const row = info.row.original;
          if (row.type !== "custom") return null;
          return (
            <button
              className="btn btn-ghost btn-sm btn-icon"
              title="Delete benchmark"
              onClick={(e) => { e.stopPropagation(); setPendingDelete(row.name); }}
            >
              <Trash2 size={14} />
            </button>
          );
        },
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
  });

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
          {(["all", "built-in", "custom"] as const).map((f) => (
            <button
              key={f}
              className={`registry-filter-chip ${typeFilter === f ? "registry-filter-chip--active" : ""}`}
              onClick={() => setTypeFilter(f)}
            >
              {f === "all" ? "All" : f === "built-in" ? "Built-in" : "Custom"}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <button className="btn" onClick={() => setShowCreate(true)}>
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
              <tr key={row.id}>
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

      {showCreate && (
        <CreateBenchmarkModal
          onCreated={() => { setShowCreate(false); loadCustom().catch(console.error); }}
          onClose={() => setShowCreate(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Benchmark"
          itemName={pendingDelete}
          description="This will permanently remove this custom benchmark."
          isDeleting={deleteCmd.isRunning}
          onConfirm={() => handleDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
