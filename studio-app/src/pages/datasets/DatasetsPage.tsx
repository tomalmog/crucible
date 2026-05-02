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
import { IngestModal } from "../../components/shared/IngestModal";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { formatSize } from "../../components/shared/RegistryRow";
import { ClusterSelect } from "../../components/shared/ClusterSelect";
import { DatasetDashboard } from "./DatasetDashboard";
import { SampleInspector } from "./SampleInspector";
import { FilterForm } from "./FilterForm";
import { useCrucible } from "../../context/CrucibleContext";
import { deleteDataset } from "../../api/studioApi";
import {
  listClusters,
  listRemoteDatasets,
  deleteRemoteDataset,
  pushDatasetToCluster,
  pullDatasetFromCluster,
} from "../../api/remoteApi";
import type { ClusterConfig, RemoteDatasetInfo } from "../../types/remote";

// ── Types ────────────────────────────────────────────────────────────

type DetailTab = "overview" | "samples" | "filter";
type LocationFilter = "local" | "remote";

const DETAIL_TABS = ["overview", "samples", "filter"] as const;
const LOCATION_FILTERS: { key: LocationFilter; label: string }[] = [
  { key: "local", label: "Local" },
  { key: "remote", label: "Remote" },
];

interface DatasetRow {
  name: string;
  sizeBytes: number;
  location: "local" | "remote";
}

// ── Component ────────────────────────────────────────────────────────

export function DatasetsPage() {
  const { dataRoot, datasets, setSelectedDataset, refreshDatasets } =
    useCrucible();
  const [detailName, setDetailName] = useState<string | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [locationFilter, setLocationFilter] = useState<LocationFilter>("local");
  const [search, setSearch] = useState("");
  const [sorting, setSorting] = useState<SortingState>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showIngest, setShowIngest] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleteFiles, setDeleteFiles] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [transferring, setTransferring] = useState<Set<string>>(new Set());
  const [transferError, setTransferError] = useState<string | null>(null);

  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [selectedCluster, setSelectedCluster] = useState("");
  const [remoteDatasets, setRemoteDatasets] = useState<RemoteDatasetInfo[]>([]);
  const [loadingRemote, setLoadingRemote] = useState(false);

  useEffect(() => {
    listClusters(dataRoot)
      .then((c) => {
        setClusters(c);
        if (c.length > 0) setSelectedCluster(c[0].name);
      })
      .catch(console.error);
  }, [dataRoot]);

  const fetchRemote = useCallback(
    async (bypassCache?: boolean) => {
      if (!selectedCluster) return;
      setLoadingRemote(true);
      try {
        setRemoteDatasets(
          await listRemoteDatasets(dataRoot, selectedCluster, bypassCache),
        );
      } catch {
        setRemoteDatasets([]);
      } finally {
        setLoadingRemote(false);
      }
    },
    [dataRoot, selectedCluster],
  );

  useEffect(() => {
    if (locationFilter === "remote" && selectedCluster) {
      fetchRemote().catch(console.error);
    }
  }, [locationFilter, selectedCluster, fetchRemote]);

  // ── Merged data ──────────────────────────────────────────────────

  const remoteNameSet = useMemo(
    () => new Set(remoteDatasets.map((d) => d.name)),
    [remoteDatasets],
  );

  const localNameSet = useMemo(
    () => new Set(datasets.map((d) => d.name)),
    [datasets],
  );

  const data = useMemo(() => {
    let rows: DatasetRow[];

    if (locationFilter === "local") {
      rows = datasets.map((d) => ({
        name: d.name,
        sizeBytes: d.sizeBytes,
        location: "local" as const,
      }));
    } else {
      rows = remoteDatasets.map((d) => ({
        name: d.name,
        sizeBytes: d.sizeBytes,
        location: "remote" as const,
      }));
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter((r) => r.name.toLowerCase().includes(q));
    }

    return rows;
  }, [datasets, remoteDatasets, locationFilter, search]);

  // ── Handlers ─────────────────────────────────────────────────────

  function handleSelect(name: string) {
    setSelectedDataset(name);
    setDetailName(name);
    setTab("overview");
  }

  function handleBack() {
    setDetailName(null);
  }

  const pendingDeleteIsLocal = useMemo(() => {
    if (!pendingDelete) return false;
    return localNameSet.has(pendingDelete);
  }, [pendingDelete, localNameSet]);

  async function handleDeleteDataset(): Promise<void> {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      const isLocal = localNameSet.has(pendingDelete);
      const isRemote = remoteNameSet.has(pendingDelete);
      if (isLocal) {
        await deleteDataset(dataRoot, pendingDelete, deleteFiles);
        await refreshDatasets();
      }
      if (isRemote && selectedCluster) {
        await deleteRemoteDataset(dataRoot, selectedCluster, pendingDelete);
        await fetchRemote();
      }
      if (!isLocal && !isRemote) {
        // Shouldn't happen but handle gracefully
        await refreshDatasets();
      }
      if (detailName === pendingDelete) setDetailName(null);
    } finally {
      setPendingDelete(null);
      setDeleteFiles(false);
      setDeleting(false);
    }
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    try {
      await refreshDatasets();
      if (selectedCluster) await fetchRemote(true);
    } finally {
      setIsRefreshing(false);
    }
  }

  async function handlePushDataset(name: string): Promise<void> {
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      await pushDatasetToCluster(dataRoot, selectedCluster, name);
      await Promise.all([refreshDatasets(), fetchRemote()]);
    } catch (err) {
      setTransferError(
        `Push "${name}" failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setTransferring((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  }

  async function handlePullDataset(name: string): Promise<void> {
    setTransferError(null);
    setTransferring((prev) => new Set(prev).add(name));
    try {
      await pullDatasetFromCluster(dataRoot, selectedCluster, name);
      await refreshDatasets();
      await fetchRemote();
    } catch (err) {
      setTransferError(
        `Pull "${name}" failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setTransferring((prev) => {
        const next = new Set(prev);
        next.delete(name);
        return next;
      });
    }
  }

  // ── Detail view ──────────────────────────────────────────────────

  if (detailName) {
    return (
      <DetailPage title={detailName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <DatasetDashboard />}
        {tab === "samples" && <SampleInspector />}
        {tab === "filter" && <FilterForm />}
      </DetailPage>
    );
  }

  // ── Table columns ────────────────────────────────────────────────

  const columns = useMemo(() => {
    const c = createColumnHelper<DatasetRow>();
    return [
      c.accessor("name", {
        header: "Name",
        cell: (info) => (
          <span className="registry-table-name">{info.getValue()}</span>
        ),
      }),
      c.accessor("sizeBytes", {
        header: "Size",
        cell: (info) => formatSize(info.getValue()),
      }),
      c.accessor("location", {
        header: "Location",
        enableSorting: false,
        cell: (info) => (
          <span className="text-muted">
            {info.getValue() === "local" ? "Local" : "Remote"}
          </span>
        ),
      }),
      c.display({
        id: "actions",
        header: "",
        cell: (info) => {
          const row = info.row.original;
          const isBusy = transferring.has(row.name);
          return (
            <div
              style={{ display: "flex", gap: 4, justifyContent: "flex-end" }}
              onClick={(e) => e.stopPropagation()}
            >
              {(row.location === "local") && (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  title="Push to remote cluster"
                  disabled={clusters.length === 0 || !selectedCluster || isBusy}
                  onClick={() =>
                    handlePushDataset(row.name).catch(console.error)
                  }
                >
                  {isBusy ? (
                    <Loader2 size={14} className="spin" />
                  ) : (
                    <Upload size={14} />
                  )}
                </button>
              )}
              {(row.location === "remote") && (
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  title="Pull to local"
                  disabled={isBusy}
                  onClick={() =>
                    handlePullDataset(row.name).catch(console.error)
                  }
                >
                  {isBusy ? (
                    <Loader2 size={14} className="spin" />
                  ) : (
                    <Download size={14} />
                  )}
                </button>
              )}
              <button
                className="btn btn-ghost btn-sm btn-icon"
                title="Delete dataset"
                onClick={() => setPendingDelete(row.name)}
              >
                <Trash2 size={14} />
              </button>
            </div>
          );
        },
      }),
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transferring, clusters, selectedCluster]);

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

  // ── List view ──────────────────────────────────────────────────

  const showClusterSelect =
    locationFilter === "remote" && clusters.length > 0;

  const showRemotePrompt =
    locationFilter === "remote" && clusters.length === 0;

  return (
    <>
      <PageHeader title="Datasets" />

      <div className="registry-toolbar">
        <div className="registry-search">
          <Search size={14} />
          <input
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            placeholder="Search datasets..."
          />
        </div>
        <div className="registry-filters">
          {LOCATION_FILTERS.map((f) => (
            <button
              key={f.key}
              className={`registry-filter-chip ${locationFilter === f.key ? "registry-filter-chip--active" : ""}`}
              onClick={() => setLocationFilter(f.key)}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        {showClusterSelect && (
          <ClusterSelect
            clusters={clusters}
            value={selectedCluster}
            onChange={setSelectedCluster}
          />
        )}
        <button
          className="btn"
          onClick={() => handleRefresh().catch(console.error)}
          disabled={isRefreshing}
        >
          <RefreshCw size={14} /> {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
        <button className="btn" onClick={() => setShowIngest(true)}>
          <Plus size={14} /> Add Dataset
        </button>
      </div>

      {transferError && (
        <p className="error-text" style={{ marginBottom: 8 }}>
          {transferError}
        </p>
      )}

      {showRemotePrompt ? (
        <p className="text-muted" style={{ padding: 24, textAlign: "center" }}>
          Add a cluster in Settings to see remote datasets.
        </p>
      ) : (
        <>
          <div className="registry-table-wrap">
            <table className="registry-table">
              <thead>
                {table.getHeaderGroups().map((hg) => (
                  <tr key={hg.id}>
                    {hg.headers.map((header) => (
                      <th
                        key={header.id}
                        onClick={header.column.getToggleSortingHandler()}
                        className={
                          header.column.getCanSort() ? "sortable" : ""
                        }
                      >
                        <div className="registry-th-content">
                          {flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
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
                {loadingRemote && locationFilter !== "local" ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: "center", padding: 24 }}>
                      <span className="text-muted">Loading...</span>
                    </td>
                  </tr>
                ) : table.getRowModel().rows.length === 0 ? (
                  <tr>
                    <td colSpan={4} style={{ textAlign: "center", padding: 24 }}>
                      <span className="text-muted">No datasets found.</span>
                    </td>
                  </tr>
                ) : (
                  table.getRowModel().rows.map((row) => (
                    <tr
                      key={row.id}
                      onClick={() => handleSelect(row.original.name)}
                      style={{ cursor: "pointer" }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <td key={cell.id}>
                          {flexRender(
                            cell.column.columnDef.cell,
                            cell.getContext(),
                          )}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {table.getPageCount() > 1 && (
            <div className="dataset-pagination">
              <span className="text-muted text-sm">
                Showing{" "}
                {table.getState().pagination.pageIndex * 15 + 1}
                &ndash;
                {Math.min(
                  (table.getState().pagination.pageIndex + 1) * 15,
                  data.length,
                )}{" "}
                of {data.length}
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
                  Page {table.getState().pagination.pageIndex + 1} of{" "}
                  {table.getPageCount()}
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
        </>
      )}

      {showIngest && <IngestModal onClose={() => setShowIngest(false)} />}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Dataset"
          itemName={pendingDelete}
          description="This will remove the dataset from the registry."
          checkbox={
            pendingDeleteIsLocal
              ? {
                  label: "Also delete data files from disk",
                  checked: deleteFiles,
                  onChange: setDeleteFiles,
                }
              : undefined
          }
          isDeleting={deleting}
          onConfirm={() => handleDeleteDataset().catch(console.error)}
          onCancel={() => {
            setPendingDelete(null);
            setDeleteFiles(false);
          }}
        />
      )}
    </>
  );
}
