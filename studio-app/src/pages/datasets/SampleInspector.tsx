import { useCallback, useEffect, useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
} from "@tanstack/react-table";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { sampleRecords, getDatasetRecordCount } from "../../api/studioApi";
import { EmptyState } from "../../components/shared/EmptyState";
import type { RecordSample } from "../../types";

const PAGE_SIZE = 25;
const columnHelper = createColumnHelper<RecordSample>();

/** Truncate text for table cells, full text shown on expand */
function truncate(text: string, max = 120): string {
  return text.length > max ? text.slice(0, max) + "…" : text;
}

export function SampleInspector() {
  const { dataRoot, selectedDataset } = useCrucible();
  const [rows, setRows] = useState<RecordSample[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchPage = useCallback(async (pageNum: number) => {
    if (!selectedDataset) return;
    setLoading(true);
    try {
      const data = await sampleRecords(dataRoot, selectedDataset, pageNum * PAGE_SIZE, PAGE_SIZE);
      setRows(data);
    } catch {
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [dataRoot, selectedDataset]);

  // Fetch record count and first page when dataset changes
  useEffect(() => {
    if (!selectedDataset) return;
    setPage(0);
    setExpandedId(null);
    getDatasetRecordCount(dataRoot, selectedDataset)
      .then(setTotalCount)
      .catch(() => setTotalCount(0));
    fetchPage(0).catch(console.error);
  }, [dataRoot, selectedDataset, fetchPage]);

  // Fetch on page change
  useEffect(() => {
    fetchPage(page).catch(console.error);
  }, [page, fetchPage]);

  // Derive columns from the data — always text, plus any extra_fields keys
  const columns = useMemo<ColumnDef<RecordSample, string>[]>(() => {
    const extraKeys = new Set<string>();
    for (const row of rows) {
      for (const k of Object.keys(row.extra_fields || {})) {
        if (k !== "quality_model") extraKeys.add(k);
      }
    }

    const cols: ColumnDef<RecordSample, string>[] = [];

    // If extra fields exist (prompt, response, etc.), show those as columns
    // instead of the flattened text field
    if (extraKeys.size > 0) {
      for (const key of Array.from(extraKeys).sort()) {
        cols.push(
          columnHelper.accessor((row) => row.extra_fields?.[key] ?? "", {
            id: key,
            header: key,
            cell: (info) => truncate(info.getValue(), 100),
          }),
        );
      }
    } else {
      // No structured fields — show raw text
      cols.push(
        columnHelper.accessor("text", {
          header: "text",
          cell: (info) => truncate(info.getValue(), 200),
        }),
      );
    }

    return cols;
  }, [rows]);

  const table = useReactTable({
    data: rows,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  if (!selectedDataset) {
    return <EmptyState title="No dataset" description="Select a dataset to view records." />;
  }

  const totalPages = Math.max(1, Math.ceil(totalCount / PAGE_SIZE));
  const start = page * PAGE_SIZE + 1;
  const end = Math.min((page + 1) * PAGE_SIZE, totalCount);

  return (
    <div className="stack">
      {loading && rows.length === 0 ? (
        <EmptyState title="Loading..." description="Fetching records." />
      ) : rows.length === 0 ? (
        <EmptyState title="No records" description="This dataset has no records." />
      ) : (
        <>
          <div className="docs-table-wrap">
            <table className="docs-table">
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
                  const isExpanded = expandedId === row.original.record_id;
                  return (
                    <>
                      <tr
                        key={row.id}
                        onClick={() => setExpandedId(isExpanded ? null : row.original.record_id)}
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
                          <td colSpan={columns.length}>
                            <div className="sample-detail">
                              {Object.keys(row.original.extra_fields || {}).length > 0 ? (
                                Object.entries(row.original.extra_fields)
                                  .filter(([k]) => k !== "quality_model")
                                  .map(([k, v]) => (
                                    <div key={k} className="sample-detail-field">
                                      <strong>{k}</strong>
                                      <pre>{v}</pre>
                                    </div>
                                  ))
                              ) : (
                                <pre>{row.original.text}</pre>
                              )}
                              <div className="sample-detail-meta">
                                <span>Quality: {row.original.quality_score.toFixed(3)}</span>
                                <span>Language: {row.original.language}</span>
                                <span className="text-mono text-xs">{row.original.record_id.slice(0, 16)}</span>
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
              Showing {start}–{end} of {totalCount.toLocaleString()} records
            </span>
            <div className="pagination-controls">
              <button
                className="btn btn-ghost btn-sm btn-icon"
                disabled={page === 0}
                onClick={() => setPage(page - 1)}
                title="Previous page"
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
                title="Next page"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
