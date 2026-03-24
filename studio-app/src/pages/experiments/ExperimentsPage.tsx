import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { EvalResultsView } from "./EvalResultsView";

interface PrefillState {
  prefill?: Record<string, unknown>;
}

export function BenchmarksPage() {
  const location = useLocation();
  const prefill = (location.state as PrefillState | null)?.prefill;
  const validPrefill = prefill?.page === "benchmarks" ? prefill : undefined;

  // Clear navigation state after reading to prevent re-apply on refresh
  if (prefill) {
    window.history.replaceState({}, "");
  }

  return (
    <>
      <PageHeader title="Benchmarks" />
      <EvalResultsView prefill={validPrefill} />
    </>
  );
}
