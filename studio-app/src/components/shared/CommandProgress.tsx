interface CommandProgressProps {
  label: string;
  percent: number;
  elapsed?: number;
  remaining?: number;
}

export function CommandProgress({ label, percent, elapsed, remaining }: CommandProgressProps) {
  return (
    <div className="progress-bar">
      <div className="progress-bar-header">
        <span className="progress-label">{label}</span>
        <span className="progress-value">
          {percent.toFixed(0)}%
        </span>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.min(100, Math.max(0, percent))}%` }}
        />
      </div>
      {(elapsed !== undefined || remaining !== undefined) && (
        <div className="progress-bar-footer">
          {elapsed !== undefined && <span>{elapsed}s elapsed</span>}
          {remaining !== undefined && <span>{remaining}s remaining</span>}
        </div>
      )}
    </div>
  );
}
