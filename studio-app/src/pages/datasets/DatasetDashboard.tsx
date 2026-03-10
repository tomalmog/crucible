import { useCrucible } from "../../context/CrucibleContext";
import { MetricCard } from "../../components/shared/MetricCard";
import { BarChart } from "../../components/shared/BarChart";
import { EmptyState } from "../../components/shared/EmptyState";

export function DatasetDashboard() {
  const { dashboard } = useCrucible();

  if (!dashboard) {
    return <EmptyState title="No dashboard data" description="Select a dataset to view quality and source composition." />;
  }

  const languageRows = Object.entries(dashboard.language_counts).map(([lang, count]) => ({
    label: lang,
    value: count,
  }));

  return (
    <div className="stack-lg">
      <div className="stats-grid">
        <MetricCard label="Dataset" value={dashboard.dataset_name} />
        <MetricCard label="Records" value={String(dashboard.record_count)} />
        <MetricCard label="Avg Quality" value={dashboard.average_quality.toFixed(3)} />
        <MetricCard label="Quality Range" value={`${dashboard.min_quality.toFixed(3)} - ${dashboard.max_quality.toFixed(3)}`} />
      </div>

      <div className="split-grid">
        {languageRows.length > 0 && (
          <div className="panel">
            <h4 className="panel-title">Language Mix</h4>
            <BarChart rows={languageRows} maxValue={dashboard.record_count} />
          </div>
        )}
      </div>
    </div>
  );
}
