import { useForge } from "../../context/ForgeContext";
import { EmptyState } from "../../components/shared/EmptyState";

export function SampleInspector() {
  const { samples } = useForge();

  if (samples.length === 0) {
    return <EmptyState title="No samples" description="Select a dataset to view sample records." />;
  }

  return (
    <div className="scroll-y">
      {samples.map((sample) => (
        <article className="sample-card" key={sample.record_id}>
          <header>
            <strong>{sample.record_id.slice(0, 14)}</strong>
            <span className="badge badge-accent">{sample.language}</span>
            <span>{sample.quality_score.toFixed(3)}</span>
          </header>
          <p>{sample.text.slice(0, 280)}...</p>
          <small>{sample.source_uri}</small>
        </article>
      ))}
    </div>
  );
}
