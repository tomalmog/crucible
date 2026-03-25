import { useNavigate } from "react-router";
import type { CommandTaskStatus } from "../../types";

interface ActivityPanelProps {
  localJobs: CommandTaskStatus[];
}

export function ActivityPanel({ localJobs }: ActivityPanelProps) {
  const navigate = useNavigate();
  const activeLocal = localJobs.filter((j) => j.status === "running");
  const totalActive = activeLocal.length;

  return (
    <div className="resource-card">
      <div className="resource-card-header">
        <h3 className="resource-card-title">Activity</h3>
        {totalActive > 0 && (
          <span className="badge badge-accent">{totalActive} active</span>
        )}
      </div>

      {totalActive === 0 ? (
        <p className="text-sm text-tertiary" style={{ margin: 0 }}>
          No active jobs.
        </p>
      ) : (
        <div>
          {activeLocal.map((j) => (
            <div key={j.task_id} className="activity-job">
              <span className="job-status-dot pulsing" style={{ background: "var(--accent)" }} />
              <span className="activity-job-name">{j.label || j.command}</span>
              <span className="activity-job-meta">{formatElapsed(j.elapsed_seconds)}</span>
            </div>
          ))}
        </div>
      )}

      <button className="btn btn-sm" onClick={() => navigate("/jobs")} style={{ justifySelf: "start" }}>
        View All Jobs
      </button>
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
