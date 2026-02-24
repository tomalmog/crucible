import { EmptyState } from "../../components/shared/EmptyState";

interface HardwareProfileViewProps {
  hardwareProfile: Record<string, string> | null;
  onRefresh: () => void;
}

export function HardwareProfileView({ hardwareProfile, onRefresh }: HardwareProfileViewProps) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Hardware Profile</h3>
        <button className="btn btn-sm" onClick={onRefresh}>Refresh</button>
      </div>
      {hardwareProfile ? (
        <dl className="runtime-key-value-list">
          {Object.entries(hardwareProfile).map(([key, value]) => (
            <div key={key} className="runtime-key-value-row">
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <EmptyState title="No hardware profile" description="Could not load hardware information." />
      )}
    </div>
  );
}
