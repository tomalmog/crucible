import { useCallback, useEffect, useState } from "react";
import { Download, Check, Loader, Server, Monitor, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
import { formatBytes } from "../../pages/hub/hubUtils";

type DownloadStatus = "idle" | "downloading" | "done" | "error";
type Destination = "local" | "remote";
type DownloadKind = "model" | "dataset";

interface DownloadModalProps {
  repoId: string;
  targetDir: string;
  size?: number;
  kind?: DownloadKind;
  onComplete: () => void;
  onClose: () => void;
}

export function DownloadModal({ repoId, targetDir, size, kind = "model", onComplete, onClose }: DownloadModalProps) {
  const { dataRoot } = useCrucible();
  const cmd = useCrucibleCommand();
  const [dest, setDest] = useState<Destination>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [cluster, setCluster] = useState("");
  const [registryName, setRegistryName] = useState("");
  const [dlStatus, setDlStatus] = useState<DownloadStatus>("idle");
  const [statusMsg, setStatusMsg] = useState("");

  useEffect(() => {
    if (dataRoot) {
      listClusters(dataRoot).then((c) => {
        setClusters(c);
        if (c.length > 0 && !cluster) setCluster(c[0].name);
      }).catch(() => setClusters([]));
    }
  }, [dataRoot]);

  // Close on Escape
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape" && dlStatus !== "downloading") onClose();
    }
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [dlStatus, onClose]);

  const handleDownload = useCallback(async () => {
    if (!dataRoot) return;
    setDlStatus("downloading");
    setStatusMsg(dest === "local" ? "Downloading..." : "Connecting to cluster...");
    try {
      const nameFlag = kind === "model" ? "--model-name" : "--dataset-name";
      const effectiveName = registryName.trim() || repoId;
      const registerArgs = dest === "local" || kind === "model"
        ? ["--register", nameFlag, effectiveName]
        : [];
      const localCmd = kind === "model" ? "download-model" : "download-dataset";
      const remoteCmd = kind === "model" ? "download-model-remote" : "download-dataset-remote";
      const args = dest === "local"
        ? ["hub", localCmd, repoId, "--target-dir", targetDir, ...registerArgs]
        : ["hub", remoteCmd, repoId, "--cluster", cluster, ...registerArgs];

      const status = await cmd.run(dataRoot, args);

      if (dest === "remote" && status.stdout) {
        const lines = status.stdout.split("\n").filter((l: string) => l.startsWith("DOWNLOAD_REMOTE: "));
        if (lines.length > 0) setStatusMsg(lines[lines.length - 1].replace("DOWNLOAD_REMOTE: ", ""));
      }

      const success = status.status === "completed";
      setDlStatus(success ? "done" : "error");
      if (!success) setStatusMsg(status.stderr?.slice(0, 300) || "Download failed");
      if (success) {
        setStatusMsg("Complete!");
        onComplete();
      }
    } catch {
      setDlStatus("error");
      setStatusMsg("Download failed");
    }
  }, [dataRoot, dest, repoId, targetDir, cluster, registryName, kind, cmd, onComplete]);

  const sizeLabel = size ? formatBytes(size) : "";
  const canClose = dlStatus !== "downloading";

  return (
    <div className="modal-backdrop" onClick={canClose ? onClose : undefined}>
      <div className="download-modal" onClick={(e) => e.stopPropagation()}>
        <div className="download-modal-header">
          <div>
            <h3 className="download-modal-title">Download {kind === "model" ? "Model" : "Dataset"}</h3>
            <div className="download-modal-repo">
              {repoId}
              {sizeLabel && <span className="download-modal-size">{sizeLabel}</span>}
            </div>
          </div>
          {canClose && (
            <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
              <X size={16} />
            </button>
          )}
        </div>

        <div className="download-modal-tabs">
          <button
            className={`download-modal-tab ${dest === "local" ? "active" : ""}`}
            onClick={() => setDest("local")}
          >
            <Monitor size={14} /> Local Machine
          </button>
          <button
            className={`download-modal-tab ${dest === "remote" ? "active" : ""}`}
            onClick={() => setDest("remote")}
          >
            <Server size={14} /> Remote Cluster
          </button>
        </div>

        <div className="download-modal-body">
          {dest === "local" && (
            <div className="download-modal-section">
              <label>
                <span>Download To</span>
                <input value={targetDir} disabled />
              </label>
              <label>
                <span>{kind === "model" ? "Model" : "Dataset"} Name</span>
                <input
                  value={registryName}
                  onChange={(e) => setRegistryName(e.currentTarget.value)}
                  placeholder={repoId}
                />
              </label>
            </div>
          )}

          {dest === "remote" && (
            <div className="download-modal-section">
              {clusters.length === 0 ? (
                <p className="text-tertiary text-xs">
                  No clusters registered. Register a cluster first in the Clusters page.
                </p>
              ) : (
                <>
                  <label>
                    <span>Cluster</span>
                    <select value={cluster} onChange={(e) => setCluster(e.target.value)}>
                      {clusters.map((c) => (
                        <option key={c.name} value={c.name}>{c.name}</option>
                      ))}
                    </select>
                  </label>
                  {kind === "model" && (
                    <label>
                      <span>Model Name</span>
                      <input
                        value={registryName}
                        onChange={(e) => setRegistryName(e.currentTarget.value)}
                        placeholder={repoId}
                      />
                    </label>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {statusMsg && (
          <div className={`download-modal-status ${dlStatus === "error" ? "error" : dlStatus === "done" ? "success" : ""}`}>
            {statusMsg}
          </div>
        )}

        <div className="download-modal-footer">
          {canClose && (
            <button className="btn btn-sm" onClick={onClose}>Cancel</button>
          )}
          <button
            className={`btn btn-sm ${dlStatus === "done" ? "btn-success" : "btn-primary"}`}
            onClick={dlStatus === "done" ? onClose : handleDownload}
            disabled={dlStatus === "downloading" || (dest === "remote" && clusters.length === 0)}
          >
            {dlStatus === "downloading" && <><Loader size={12} className="spin" /> Downloading...</>}
            {dlStatus === "done" && <><Check size={12} /> Done</>}
            {dlStatus === "error" && <><Download size={12} /> Retry</>}
            {dlStatus === "idle" && (
              dest === "local"
                ? <><Download size={12} /> Download Locally</>
                : <><Server size={12} /> Download to Cluster</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
