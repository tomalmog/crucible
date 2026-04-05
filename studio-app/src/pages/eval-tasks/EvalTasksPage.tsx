import { PageHeader } from "../../components/shared/PageHeader";
import { EmptyState } from "../../components/shared/EmptyState";

export function EvalTasksPage() {
  return (
    <>
      <PageHeader title="Benchmarks" />
      <EmptyState
        title="Coming soon"
        description="Create and manage custom evaluation benchmarks. Define question/answer pairs to test your models against."
      />
    </>
  );
}
