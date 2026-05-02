import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import {
  ArrowUp,
  ArrowDown,
  Plus,
  Search,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Upload,
  Download,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { RegisterModelModal } from "../../components/shared/RegisterModelModal";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { ClusterSelect } from "../../components/shared/ClusterSelect";
import { formatSize } from "../../components/shared/RegistryRow";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import { listClusters, getRemoteModelSizes } from "../../api/remoteApi";
import { ModelOverview } from "./ModelOverview";
import { ModelMergeForm } from "./ModelMergeForm";
import type { ModelEntry } from "../../types/models";
import type { ClusterConfig } from "../../types/remote";

// ── Types ────────────────────────────────────────────────────────────

type DetailTab = "overview" | "merge";
type LocationFilter = "local" | "remote";
const DETAIL_TABS = ["overview", "merge"] as const;

const POLL_MS = 400;
const PAGE_SIZE = 15;

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function locationLabel(m: ModelEntry): string {
  return m.hasLocal ? "Local" : "Remote";
}

// ── Component ────────────────────────────────────────────────────────

export function ModelsPage() {
  const { dataRoot, models, setSelectedModel, refreshModels } = useCrucible();
  const command = useCrucibleCommand();

  // Detail view state
  const [detailEntry, setDetailEntry] = useState<ModelEntry | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");

  // Table state
  const [sorting, setSorting] = useState<SortingState>([]);
  const [search, setSearch] = useState("");
  const [locationFilter, setLocationFilter] = useState<LocationFilter>("local");
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Modals
  const [showRegister, setShowRegister] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);
  const [deleteLocalFiles, setDeleteLocalFiles] = useState(true);

  // Cluster & remote state
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteSizes, setRemoteSizes] = useState<Record<string, number>>({});
  const [transferring, setTransferring] = useState<Set<string>>(new Set());
  const [transferError, setTransferError] = useState<string | null>(null);

  // ── Cluster loading ──────────────────────────────────────────────

  useEffect(() => {
    listClusters(dataRoot)
      .then((c) => {
        setClusters(c);
        if (c.length > 0) setSelectedCluster(c[0].name);
      })
      .catch(console.error);
  }, [dataRoot]);

  const clusterHost = useMemo(
    () => clusters.find((c) => c.name === selectedCluster)?.host ?? selectedCluster,
    [clusters, selectedCluster],
  );

  const fetchRemoteSizes = useCallback(
    async (bypassCache?: boolean) => {
      if (!selectedCluster) return;
      try {
        const sizes = await getRemoteModelSizes(dataRoot, selectedCluster, bypassCache);
        setRemoteSizes(sizes);
      } catch {
        setRemoteSizes({});
      }
    },
    [dataRoot, selectedCluster],
  );

  useEffect(() => {
    if (locationFilter === "remote" && selectedCluster) {
      fetchRemoteSizes().catch(console.error);
    }
  }, [locationFilter, selectedCluster, fetchRemoteSizes]);

  // ── Transfer helpers ─────────────────────────────────────────────

  const pollUntilDone = useCallback(async (taskId: string) => {
    while (true) {
      const s = await getCrucibleCommandStatus(taskId);
      if (s.status !== "running") return s;
      await new Promise((r) => setTimeout(r, POLL_MS));
    }
  }, []);

  async function handlePushModel(entry: ModelEntry): Promise<void> {
    const name = entry.modelName;
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "remote", "push-model", "--cluster", selectedCluster, "--name", name,
      ]);
      const status = await pollUntilDone(task_id);
      if (status.status === "failed") throw new Error(status.stderr || "Push failed");
      await refreshModels();
    } catch (err) {
      setTransferError(`Push "${name}" failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTransferring((prev) => { const n = new Set(prev); n.delete(name); return n; });
    }
  }

  async function handlePullModel(entry: ModelEntry): Promise<void> {
    const name = entry.modelName;
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      const { task_id } = await startCrucibleCommand(dataRoot, [
        "model", "pull", "--name", name,
      ]);
      const status = await pollUntilDone(task_id);
      if (status.status === "failed") throw new Error(status.stderr || "Pull failed");
      await refreshModels();
    } catch (err) {
      setTransferError(`Pull "${name}" failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTransferring((prev) => { const n = new Set(prev); n.delete(name); return n; });
    }
  }

  // ── Actions ──────────────────────────────────────────────────────

  function handleSelect(entry: ModelEntry) {
    setSelectedModel(entry);
    setDetailEntry(entry);
    setTab("overview");
  }

  function handleBack() {
    setDetailEntry(null);
  }

  async function handleDelete(): Promise<void> {
    if (!pendingDelete) return;
    const args = ["model", "delete", "--name", pendingDelete.modelName, "--yes"];
    if (deleteLocalFiles && pendingDelete.hasLocal) args.push("--delete-local");
    if (pendingDelete.hasRemote) args.push("--include-remote");
    await command.run(dataRoot, args);
    setPendingDelete(null);
    if (detailEntry?.modelName === pendingDelete.modelName) setDetailEntry(null);
    await refreshModels();
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshModels();
    if (locationFilter === "remote") {
      await fetchRemoteSizes(true);
    }
    setIsRefreshing(false);
  }

  // ── Filtered data ────────────────────────────────────────────────

  const data = useMemo(() => {
    let rows = locationFilter === "local"
      ? models.filter((m) => m.hasLocal)
      : models.filter((m) => m.hasRemote && (!selectedCluster || m.remoteHost === clusterHost));
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((r) => r.modelName.toLowerCase().includes(q));
    }
    return rows;
  }, [models, locationFilter, selectedCluster, clusterHost, search]);

  // ── Table columns ────────────────────────────────────────────────

  const columns = useMemo(() => {
    const c = createColumnHelper<ModelEntry>();
    return [
      c.accessor("modelName", {
        header: "Name",
        cell: (info) => <span className="registry-table-name">{info.getValue()}</span>,
      }),
      c.accessor("sizeBytes", {
        header: "Size",
        cell: (info) => {
          const m = info.row.original;
          const bytes = m.hasRemote && remoteSizes[m.modelName]
            ? remoteSizes[m.modelName]
            : m.sizeBytes;
          return <span className="text-muted">{formatSize(bytes)}</span>;
        },
      }),
      c.accessor("createdAt", {
        header: "Created",
        cell: (info) => {
          const v = info.getValue();
          return <span className="text-muted">{v ? formatDate(v) : "\u2014"}</span>;
        },
      }),
      c.display({
        id: "location",
        header: "Location",
        cell: (info) => (
          <span className="text-muted">{locationLabel(info.row.original)}</span>
        ),
      }),
      c.display({
        id: "actions",
        header: "",
        cell: (info) => {
          const m = info.row.original;
          return (
            <div style={{ display: "flex", gap: 4, justifyContent: "flex-end" }}>
              {m.hasLocal && clusters.length > 0 && (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  title="Push to remote cluster"
                  disabled={transferring.has(m.modelName)}
                  onClick={(e) => { e.stopPropagation(); handlePushModel(m).catch(console.error); }}
                >
                  {transferring.has(m.modelName) ? <Loader2 size={14} className="spin" /> : <Upload size={14} />}
                </button>
              )}
              {m.hasRemote && !m.hasLocal && (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  title="Pull to local"
                  disabled={transferring.has(m.modelName)}
                  onClick={(e) => { e.stopPropagation(); handlePullModel(m).catch(console.error); }}
                >
                  {transferring.has(m.modelName) ? <Loader2 size={14} className="spin" /> : <Download size={14} />}
                </button>
              )}
              <button
                className="btn btn-ghost btn-sm btn-icon"
                title="Delete model"
                onClick={(e) => { e.stopPropagation(); setDeleteLocalFiles(true); setPendingDelete(m); }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        },
      }),
    ];
  }, [clusters, transferring, remoteSizes]);

  // ── Table instance ───────────────────────────────────────────────

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: PAGE_SIZE } },
  });

  // ── Detail view ──────────────────────────────────────────────────

  if (detailEntry) {
    return (
      <DetailPage title={detailEntry.modelName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <ModelOverview entry={detailEntry} />}
        {tab === "merge" && <ModelMergeForm />}
      </DetailPage>
    );
  }

  // ── List view ────────────────────────────────────────────────────

  const pageIndex = table.getState().pagination.pageIndex;
  const totalRows = data.length;
  const start = pageIndex * PAGE_SIZE + 1;
  const end = Math.min((pageIndex + 1) * PAGE_SIZE, totalRows);

  return (
    <>
      <PageHeader title="Models" />

      <div className="registry-toolbar">
        <div className="registry-search">
          <Search size={14} />
          <input
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            placeholder="Search models..."
          />
        </div>
        <div className="registry-filters">
          {(["local", "remote"] as const).map((f) => (
            <button
              key={f}
              className={`registry-filter-chip ${locationFilter === f ? "registry-filter-chip--active" : ""}`}
              onClick={() => setLocationFilter(f)}
            >
              {f === "local" ? "Local" : "Remote"}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        {locationFilter === "remote" && clusters.length > 0 && (
          <ClusterSelect clusters={clusters} value={selectedCluster} onChange={setSelectedCluster} />
        )}
        <button
          className="btn"
          onClick={() => handleRefresh().catch(console.error)}
          disabled={isRefreshing}
        >
          <RefreshCw size={14} /> Refresh
        </button>
        <button className="btn" onClick={() => setShowRegister(true)}>
          <Plus size={14} /> Add Model
        </button>
      </div>

      {transferError && (
        <p className="error-text" style={{ marginBottom: 8 }}>{transferError}</p>
      )}

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
                onClick={() => handleSelect(row.original)}
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
            Showing {start}&ndash;{end} of {totalRows}
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
              Page {pageIndex + 1} of {table.getPageCount()}
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

      {showRegister && (
        <RegisterModelModal
          onComplete={() => setShowRegister(false)}
          onClose={() => setShowRegister(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Model"
          itemName={pendingDelete.modelName}
          isDeleting={command.isRunning}
          checkbox={pendingDelete.hasLocal ? {
            label: "Delete files from disk",
            checked: deleteLocalFiles,
            onChange: setDeleteLocalFiles,
          } : undefined}
          onConfirm={() => handleDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
