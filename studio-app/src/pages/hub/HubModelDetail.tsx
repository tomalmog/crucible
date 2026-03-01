import { useState, useEffect } from "react";
import { ArrowLeft, Download, Check, Loader } from "lucide-react";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { useForge } from "../../context/ForgeContext";
import { HubModelDetail as ModelDetail } from "./hubTypes";
import { formatBytes, formatCount, formatDate } from "./hubUtils";

type DownloadStatus = "idle" | "downloading" | "done" | "error";

interface Props {
  repoId: string;
  targetDir: string;
  onBack: () => void;
}

export function HubModelDetail({ repoId, targetDir, onBack }: Props) {
  const { dataRoot, refreshModels } = useForge();
  const infoCmd = useForgeCommand();
  const downloadCmd = useForgeCommand();
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [dlState, setDlState] = useState<DownloadStatus>("idle");

  useEffect(() => {
    if (!dataRoot) return;
    infoCmd.run(dataRoot, ["hub", "model-info", repoId, "--json"]).then((s) => {
      if (s.status === "completed" && s.stdout) setDetail(JSON.parse(s.stdout));
    }).catch(console.error);
  }, [dataRoot, repoId]);

  async function handleDownload() {
    if (!dataRoot) return;
    setDlState("downloading");
    try {
      const s = await downloadCmd.run(dataRoot, ["hub", "download-model", repoId, "--target-dir", targetDir]);
      const success = s.status === "completed";
      setDlState(success ? "done" : "error");
      if (success) refreshModels().catch(console.error);
    } catch { setDlState("error"); }
  }

  return (
    <div className="hub-detail">
      <div className="hub-detail-topbar">
        <button className="btn btn-ghost btn-sm" onClick={onBack}>
          <ArrowLeft size={14} /> Back to results
        </button>
        {detail && (
          <button
            className={`btn btn-sm ${dlState === "done" ? "btn-success" : dlState === "error" ? "btn-error" : "btn-primary"}`}
            onClick={handleDownload}
            disabled={dlState === "downloading"}
          >
            {dlState === "downloading" && <><Loader size={12} className="spin" /> Downloading...</>}
            {dlState === "done" && <><Check size={12} /> Downloaded</>}
            {dlState === "error" && <><Download size={12} /> Retry</>}
            {dlState === "idle" && <><Download size={12} /> Download ({formatBytes(detail.total_size)})</>}
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
                {detail.library && (
                  <div className="hub-detail-kv-row">
                    <dt>Library</dt>
                    <dd>{detail.library}</dd>
                  </div>
                )}
                {detail.task && (
                  <div className="hub-detail-kv-row">
                    <dt>Task</dt>
                    <dd>{detail.task}</dd>
                  </div>
                )}
                {detail.base_model && (
                  <div className="hub-detail-kv-row">
                    <dt>Base model</dt>
                    <dd>{detail.base_model}</dd>
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

          <div className="hub-detail-files">
            <div className="hub-detail-files-heading">Files ({detail.files.length})</div>
            <div className="hub-detail-file-list">
              {[...detail.files]
                .sort((a, b) => b.size - a.size)
                .map((f) => (
                  <div className="hub-detail-file-row" key={f.filename}>
                    <span className="hub-detail-file-name">{f.filename}</span>
                    <span className="hub-detail-file-size">{formatBytes(f.size)}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
