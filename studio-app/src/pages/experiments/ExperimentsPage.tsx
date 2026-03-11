import { PageHeader } from "../../components/shared/PageHeader";
import { EvalResultsView } from "./EvalResultsView";

export function BenchmarksPage() {
  return (
    <>
      <PageHeader title="Benchmarks" />
      <EvalResultsView />
    </>
  );
}
