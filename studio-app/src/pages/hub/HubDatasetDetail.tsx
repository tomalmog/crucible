import { useState, useEffect, useMemo } from "react";
import { ArrowLeft, Download, ChevronDown } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { DownloadModal } from "../../components/shared/DownloadModal";
import { HubDatasetDetail as DatasetDetail, HubFileEntry } from "./hubTypes";
import { formatBytes, formatCount, formatDate } from "./hubUtils";

interface Props {
  repoId: string;
  targetDir: string;
  onBack: () => void;
}

const PAGE_SIZE = 20;

function FileList({ files, visibleCount, onShowMore }: {
  files: HubFileEntry[];
  visibleCount: number;
  onShowMore: () => void;
}) {
  const sorted = useMemo(() => [...files].sort((a, b) => b.size - a.size), [files]);
  const visible = sorted.slice(0, visibleCount);
  const remaining = files.length - visibleCount;

  return (
    <div className="hub-detail-files">
      <div className="hub-detail-files-heading">
        Largest files ({visibleCount < files.length ? `${visibleCount} of ${files.length}` : files.length})
      </div>
      <div className="hub-detail-file-list">
        {visible.map((f) => (
          <div className="hub-detail-file-row" key={f.filename}>
            <span className="hub-detail-file-name">{f.filename}</span>
            <span className="hub-detail-file-size">{formatBytes(f.size)}</span>
          </div>
        ))}
      </div>
      {remaining > 0 && (
        <button className="btn btn-ghost btn-sm hub-detail-show-more" onClick={onShowMore}>
          <ChevronDown size={12} />
          Show {Math.min(remaining, PAGE_SIZE)} more files
        </button>
      )}
    </div>
  );
}

export function HubDatasetDetail({ repoId, targetDir, onBack }: Props) {
  const { dataRoot, refreshDatasets } = useForge();
  const infoCmd = useForgeCommand();
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [showDownload, setShowDownload] = useState(false);
  const [fileCount, setFileCount] = useState(PAGE_SIZE);

  // Fetch dataset metadata when the detail view mounts
  useEffect(() => {
    if (!dataRoot) return;
    infoCmd.run(dataRoot, ["hub", "dataset-info", repoId, "--json"]).then((s) => {
      if (s.status === "completed" && s.stdout) setDetail(JSON.parse(s.stdout));
    }).catch(console.error);
  }, [dataRoot, repoId]);

  return (
    <div className="hub-detail">
      <div className="hub-detail-topbar">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <ArrowLeft size={14} /> Back to results
        </button>
        {detail && (
          <button className="btn btn-sm btn-primary" onClick={() => setShowDownload(true)}>
            <Download size={12} /> Download ({formatBytes(detail.total_size)})
          </button>
        )}
      </div>

      {infoCmd.isRunning && !detail && (
        <p className="text-tertiary text-xs">Loading details...</p>
      )}
      {infoCmd.error && <p className="error-text">{infoCmd.error}</p>}

      {detail && (
        <div className="hub-detail-body">
          <div className="hub-detail-sidebar">
            <div className="hub-detail-title">{detail.repo_id}</div>
            {detail.author && <div className="hub-detail-author">{detail.author}</div>}

            <div className="hub-detail-meta">
              <dl className="hub-detail-kv">
                <div className="hub-detail-kv-row">
                  <dt>Size</dt>
                  <dd>{formatBytes(detail.total_size)}</dd>
                </div>
                <div className="hub-detail-kv-row">
                  <dt>Downloads</dt>
                  <dd>{formatCount(detail.downloads)}</dd>
                </div>
                <div className="hub-detail-kv-row">
                  <dt>Likes</dt>
                  <dd>{formatCount(detail.likes)}</dd>
                </div>
                {detail.license && (
                  <div className="hub-detail-kv-row">
                    <dt>License</dt>
                    <dd>{detail.license}</dd>
                  </div>
                )}
                {detail.task_categories.length > 0 && (
                  <div className="hub-detail-kv-row">
                    <dt>Tasks</dt>
                    <dd>{detail.task_categories.join(", ")}</dd>
                  </div>
                )}
                {detail.gated && (
                  <div className="hub-detail-kv-row">
                    <dt>Access</dt>
                    <dd>Gated</dd>
                  </div>
                )}
                {detail.created_at && (
                  <div className="hub-detail-kv-row">
                    <dt>Created</dt>
                    <dd>{formatDate(detail.created_at)}</dd>
                  </div>
                )}
              </dl>
            </div>
          </div>

          <FileList
            files={detail.files}
            visibleCount={fileCount}
            onShowMore={() => setFileCount((c) => c + PAGE_SIZE)}
          />
        </div>
      )}

      {showDownload && detail && (
        <DownloadModal
          repoId={repoId}
          targetDir={targetDir}
          size={detail.total_size}
          kind="dataset"
          onComplete={() => refreshDatasets().catch(console.error)}
          onClose={() => setShowDownload(false)}
        />
      )}
    </div>
  );
}
