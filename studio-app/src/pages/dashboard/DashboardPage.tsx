import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import {
  Zap, FlaskConical, Globe, Activity, AlertTriangle, CheckCircle2,
  Box, Database, Server, Loader2,
} from "lucide-react";
import { useCrucible } from "../../context/CrucibleContext";
import { useUnifiedJobs } from "../../hooks/useUnifiedJobs";
import { listClusters, getClusterInfo, listRemoteDatasets } from "../../api/remoteApi";
import type { JobRecord } from "../../types/jobs";
import type { ClusterConfig } from "../../types/remote";

import { statusBadgeClass } from "../jobs/JobsPage";
import { DashboardLeaderboard } from "./DashboardLeaderboard";

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(ms / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function DashboardPage(): React.ReactNode {
  const { dataRoot, models, datasets, refreshModels, refreshDatasets } = useCrucible();
  const { jobs, refresh: refreshJobs } = useUnifiedJobs(dataRoot);
  const navigate = useNavigate();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const [loadingClusters, setLoadingClusters] = useState(false);

  const [clusters, setClusters] = useState<ClusterConfig[]>([]);
  const [connectedHosts, setConnectedHosts] = useState<Set<string>>(new Set());
  const [remoteDatasetCount, setRemoteDatasetCount] = useState(0);

  const probeRef = React.useRef(0);
  const probeClusters = React.useCallback(async () => {
    const id = ++probeRef.current;
    const cls = await listClusters(dataRoot).catch(() => [] as ClusterConfig[]);
    if (id !== probeRef.current) return;
    setClusters(cls);
    const hosts = new Set<string>();
    let dsCount = 0;
    await Promise.all(cls.map(async (c) => {
      try {
        const info = await getClusterInfo(dataRoot, c.name, true);
        if (info.isConnected) {
          hosts.add(c.host);
          const ds = await listRemoteDatasets(dataRoot, c.name, true).catch(() => []);
          dsCount += ds.length;
        }
      } catch { /* unreachable cluster */ }
    }));
    if (id === probeRef.current) {
      setConnectedHosts(hosts);
      setRemoteDatasetCount(dsCount);
    }
  }, [dataRoot]);

  // Probe on mount
  useEffect(() => { probeClusters(); }, [probeClusters]);

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    setLoadingModels(true);
    setLoadingDatasets(true);
    setLoadingJobs(true);
    setLoadingClusters(true);
    await Promise.all([
      refreshModels().finally(() => setLoadingModels(false)),
      refreshDatasets().finally(() => setLoadingDatasets(false)),
      refreshJobs().finally(() => setLoadingJobs(false)),
      probeClusters().finally(() => setLoadingClusters(false)),
    ]);
    setIsRefreshing(false);
  }

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.state === "running" || j.state === "submitting"),
    [jobs],
  );
  const recentFailed = useMemo(
    () => jobs.filter((j) => j.state === "failed").slice(0, 3),
    [jobs],
  );
  const recentCompleted = useMemo(
    () => jobs.filter((j) => j.state === "completed").slice(0, 5),
    [jobs],
  );
  const completedEvalJobs = useMemo(
    () => jobs.filter((j) => j.state === "completed" && j.jobType === "eval").slice(0, 20),
    [jobs],
  );

  const localModels = models.filter((m) => m.hasLocal).length;
  const remoteModels = models.filter(
    (m) => m.hasRemote && !m.hasLocal && connectedHosts.has(m.remoteHost),
  ).length;
  const totalModels = localModels + remoteModels;
  const totalDatasets = datasets.length + remoteDatasetCount;

  return (
    <div className="dashboard-page">
      <div className="page-header">
        <h1>Dashboard</h1>
        <div className="page-header-actions">
          <button
            className="btn"
            onClick={() => handleRefresh().catch(console.error)}
            disabled={isRefreshing}
          >
            {isRefreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* ── Quick Stats ── */}
      <div className="stats-grid" style={{ marginBottom: 20 }}>
        <StatCard
          label="Models"
          value={totalModels}
          sub={remoteModels > 0 ? `${localModels} local · ${remoteModels} remote` : undefined}
          icon={<Box size={14} />}
          loading={loadingModels}
          onClick={() => navigate("/models")}
        />
        <StatCard
          label="Datasets"
          value={totalDatasets}
          sub={remoteDatasetCount > 0 ? `${datasets.length} local · ${remoteDatasetCount} remote` : undefined}
          icon={<Database size={14} />}
          loading={loadingDatasets || loadingClusters}
          onClick={() => navigate("/datasets")}
        />
        <StatCard
          label="Clusters"
          value={clusters.length}
          icon={<Server size={14} />}
          loading={loadingClusters}
          onClick={() => navigate("/clusters")}
        />
        <StatCard
          label="Active Jobs"
          value={activeJobs.length}
          icon={<Activity size={14} />}
          loading={loadingJobs}
          onClick={() => navigate("/jobs")}
        />
      </div>

      {/* ── Active Jobs ── */}
      {(activeJobs.length > 0 || loadingJobs) && (
        <div className="resource-card" style={{ marginBottom: 16 }}>
          <div className="resource-card-header">
            <h3 className="resource-card-title">Active Jobs</h3>
            <span className="badge badge-accent">{activeJobs.length} running</span>
          </div>
          {loadingJobs ? <DashboardSpinner /> : (
            <div className="dashboard-job-list">
              {activeJobs.map((j) => (
                <JobRow key={j.jobId} job={j} onClick={() => navigate("/jobs", { state: { statusFilter: "running" } })} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Recent Failures ── */}
      {(recentFailed.length > 0 || loadingJobs) && (
        <div className="resource-card" style={{ marginBottom: 16 }}>
          <div className="resource-card-header">
            <h3 className="resource-card-title">Recent Failures</h3>
            <AlertTriangle size={14} style={{ color: "var(--error)" }} />
          </div>
          {loadingJobs ? <DashboardSpinner /> : (
            <div className="dashboard-job-list">
              {recentFailed.map((j) => (
                <JobRow key={j.jobId} job={j} onClick={() => navigate("/jobs", { state: { statusFilter: "failed" } })} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Leaderboard + Recent Completed side by side ── */}
      <div className="dashboard-two-col">
        <DashboardLeaderboard
          dataRoot={dataRoot}
          completedJobs={completedEvalJobs}
        />

        <div className="resource-card">
          <div className="resource-card-header">
            <h3 className="resource-card-title">Recent Completed</h3>
            <CheckCircle2 size={14} style={{ color: "var(--success)" }} />
          </div>
          {loadingJobs ? <DashboardSpinner /> : recentCompleted.length === 0 ? (
            <p className="text-tertiary" style={{ fontSize: "0.8125rem" }}>No completed jobs yet.</p>
          ) : (
            <div className="dashboard-job-list">
              {recentCompleted.map((j) => (
                <JobRow key={j.jobId} job={j} onClick={() => navigate("/jobs", { state: { statusFilter: "completed" } })} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Quick Actions ── */}
      <div className="resource-card" style={{ marginTop: 16 }}>
        <div className="resource-card-header">
          <h3 className="resource-card-title">Quick Actions</h3>
        </div>
        <div className="dashboard-actions">
          <button className="btn" onClick={() => navigate("/training")}>
            <Zap size={14} /> Train a Model
          </button>
          <button className="btn" onClick={() => navigate("/benchmarks")}>
            <FlaskConical size={14} /> Run Eval
          </button>
          <button className="btn" onClick={() => navigate("/hub")}>
            <Globe size={14} /> Browse Hub
          </button>
          <button className="btn" onClick={() => navigate("/jobs")}>
            <Activity size={14} /> All Jobs
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Stat card ── */

interface StatCardProps {
  label: string;
  value: number;
  sub?: string;
  icon: React.ReactNode;
  loading?: boolean;
  onClick: () => void;
}

function StatCard({ label, value, sub, icon, loading, onClick }: StatCardProps): React.ReactNode {
  return (
    <button className="metric-card metric-card-clickable" onClick={onClick}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ color: "var(--accent)" }}>{icon}</span>
        <span className="metric-label">{label}</span>
      </div>
      {loading
        ? <Loader2 size={18} className="spin" style={{ color: "var(--text-tertiary)" }} />
        : <span className="metric-value">{value}</span>
      }
      {!loading && sub && <span className="metric-sub">{sub}</span>}
    </button>
  );
}

function DashboardSpinner(): React.ReactNode {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: 16 }}>
      <Loader2 size={16} className="spin" style={{ color: "var(--text-tertiary)" }} />
    </div>
  );
}

/* ── Job row ── */

function JobRow({ job, onClick }: { job: JobRecord; onClick: () => void }): React.ReactNode {
  return (
    <button className="dashboard-job-row" onClick={onClick}>
      <div className="dashboard-job-info">
        <span className="dashboard-job-label">{job.label || job.jobType}</span>
        <span className="dashboard-job-meta">
          {job.jobType} · {timeAgo(job.updatedAt || job.createdAt)}
        </span>
      </div>
      <span className={statusBadgeClass(job.state)}>{job.state}</span>
    </button>
  );
}
