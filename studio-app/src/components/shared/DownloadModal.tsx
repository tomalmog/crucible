import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { Download, Check, Loader, Server, Monitor, X } from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { startCrucibleCommand } from "../../api/studioApi";
import { listClusters } from "../../api/remoteApi";
import type { ClusterConfig } from "../../types/remote";
import { formatBytes } from "../../pages/hub/hubUtils";

type DownloadStatus = "idle" | "downloading" | "done" | "error";
type Destination = "local" | "remote";
type DownloadKind = "model" | "dataset";

function _extractErrorMessage(stderr: string | undefined): string {
  if (!stderr) return "";
  const lines = stderr.split("\n");
  // Walk backwards to find the last CrucibleError or Python exception line
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line) continue;
    const crucible = line.match(/Crucible\w+Error:\s*(.+)/);
    if (crucible) return crucible[1];
    const pyErr = line.match(/^\w+Error:\s*(.+)/);
    if (pyErr) return line;
  }
  return stderr.slice(-300);
}

interface DownloadModalProps {
  repoId: string;
  targetDir: string;
  size?: number;
  kind?: DownloadKind;
  splits?: string[];
  onComplete: () => void;
  onClose: () => void;
}

export function DownloadModal({ repoId, targetDir, size, kind = "model", splits: splitsProp, onComplete, onClose }: DownloadModalProps) {
  const { dataRoot } = useCrucible();
  const navigate = useNavigate();
  const cmd = useCrucibleCommand();
  const splitInfoCmd = useCrucibleCommand();
  const [dest, setDest] = useState<Destination>("local");
  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [cluster, setCluster] = useState("");
  const [registryName, setRegistryName] = useState("");
  const [dlStatus, setDlStatus] = useState<DownloadStatus>("idle");
  const [statusMsg, setStatusMsg] = useState("");
  const [splitsLoaded, setSplitsLoaded] = useState(kind !== "dataset" || !!splitsProp);
  const [availableSplits, setAvailableSplits] = useState<string[]>(splitsProp ?? []);
  const [selectedSplit, setSelectedSplit] = useState(
    splitsProp?.includes("train") ? "train" : splitsProp?.[0] ?? "train",
  );

  // Fetch splits if not provided (e.g. opening download from search results)
  useEffect(() => {
    if (kind !== "dataset" || splitsProp || !dataRoot) return;
    setSplitsLoaded(false);
    splitInfoCmd.run(dataRoot, ["hub", "dataset-info", repoId, "--json"]).then((s) => {
      if (s.status === "completed" && s.stdout) {
        try {
          const info = JSON.parse(s.stdout);
          const fetched: string[] = info.splits ?? ["train"];
          setAvailableSplits(fetched);
          setSelectedSplit(fetched.includes("train") ? "train" : fetched[0]);
        } catch { /* ignore parse errors */ }
      }
      setSplitsLoaded(true);
    }).catch(() => setSplitsLoaded(true));
  }, [dataRoot, repoId, kind, splitsProp]);

  useEffect(() => {
    if (splitsProp) {
      setAvailableSplits(splitsProp);
      setSelectedSplit(splitsProp.includes("train") ? "train" : splitsProp[0]);
      setSplitsLoaded(true);
    }
  }, [splitsProp]);

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
    try {
      const effectiveName = registryName.trim() || repoId;
      const label = `Hub Download · ${effectiveName}`;

      if (dest === "remote") {
        // Remote downloads stay blocking (they're fast SSH commands)
        setDlStatus("downloading");
        setStatusMsg("Connecting to cluster...");
        const remoteCmd = kind === "model" ? "download-model-remote" : "download-dataset-remote";
        const nameFlag = kind === "model" ? "--model-name" : "--dataset-name";
        const registerArgs = kind === "model" ? ["--register", nameFlag, effectiveName] : [];
        const args = ["hub", remoteCmd, repoId, "--cluster", cluster, ...registerArgs];
        const status = await cmd.run(dataRoot, args);
        if (status.status === "completed") {
          setDlStatus("done");
          setStatusMsg("Complete!");
          onComplete();
        } else {
          setDlStatus("error");
          setStatusMsg(_extractErrorMessage(status.stderr) || "Download failed");
        }
        return;
      }

      // Local downloads: fire-and-forget as a job, close modal, go to jobs page
      const splitArgs = kind === "dataset" ? ["--split", selectedSplit] : [];
      const args = [
        "hub-download", repoId,
        "--kind", kind,
        "--target-dir", targetDir,
        ...splitArgs,
        "--register",
        kind === "model" ? "--model-name" : "--dataset-name", effectiveName,
      ];
      await startCrucibleCommand(dataRoot, args, label);
      onClose();
      navigate("/jobs", { state: { statusFilter: "running" } });
    } catch {
      setDlStatus("error");
      setStatusMsg("Download failed");
    }
  }, [dataRoot, dest, repoId, targetDir, cluster, registryName, kind, selectedSplit, cmd, onComplete, onClose, navigate]);

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
              {kind === "dataset" && availableSplits.length > 1 && (
                <label>
                  <span>Split</span>
                  <select value={selectedSplit} onChange={(e) => setSelectedSplit(e.target.value)}>
                    {availableSplits.map((s) => <option key={s} value={s}>{s}</option>)}
                  </select>
                </label>
              )}
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
            disabled={dlStatus === "downloading" || (dest === "remote" && clusters.length === 0) || (kind === "dataset" && !splitsLoaded)}
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
