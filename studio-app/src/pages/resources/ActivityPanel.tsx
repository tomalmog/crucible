import { useNavigate } from "react-router";
import { statusBadgeClass } from "../jobs/JobsPage";
import type { CommandTaskStatus } from "../../types";
import type { RemoteJobRecord } from "../../types/remote";

interface ActivityPanelProps {
  localJobs: CommandTaskStatus[];
  remoteJobs: RemoteJobRecord[];
}

export function ActivityPanel({ localJobs, remoteJobs }: ActivityPanelProps) {
  const navigate = useNavigate();
  const activeLocal = localJobs.filter((j) => j.status === "running");
  const activeRemote = remoteJobs.filter(
    (j) => j.state === "running" || j.state === "pending" || j.state === "submitting",
  );
  const totalActive = activeLocal.length + activeRemote.length;

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Activity</h3>
        {totalActive > 0 && (
          <span className="badge badge-accent">{totalActive} active</span>
        )}
      </div>
      {totalActive === 0 ? (
        <p className="text-sm text-tertiary" style={{ padding: "0 16px 16px" }}>
          No active jobs.
        </p>
      ) : (
        <div style={{ padding: "0 16px 16px" }}>
          {activeLocal.map((j) => (
            <div key={j.task_id} className="flex-row" style={{ marginBottom: 6 }}>
              <span className="text-sm" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {j.label || j.command}
              </span>
              <span className={statusBadgeClass(j.status)}>{j.status}</span>
              <span className="text-xs text-tertiary">{formatElapsed(j.elapsed_seconds)}</span>
            </div>
          ))}
          {activeRemote.map((j) => (
            <div key={j.jobId} className="flex-row" style={{ marginBottom: 6 }}>
              <span className="text-sm" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {j.trainingMethod} @ {j.clusterName}
              </span>
              <span className={statusBadgeClass(j.state)}>{j.state}</span>
            </div>
          ))}
        </div>
      )}
      <div style={{ padding: "0 16px 16px" }}>
        <button className="btn btn-sm" onClick={() => navigate("/jobs")}>
          View All Jobs
        </button>
      </div>
    </div>
  );
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
