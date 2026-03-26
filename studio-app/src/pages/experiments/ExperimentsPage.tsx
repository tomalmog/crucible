import { useState } from "react";
import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { EvalResultsView } from "./EvalResultsView";
import { EvalCompareView } from "./EvalCompareView";

type BenchTab = "evaluate" | "compare";
const TABS: readonly BenchTab[] = ["evaluate", "compare"];
const TAB_LABELS: Record<BenchTab, string> = { evaluate: "Evaluate", compare: "Compare" };

interface PrefillState {
  prefill?: Record<string, unknown>;
}

export function BenchmarksPage() {
  const location = useLocation();
  const prefill = (location.state as PrefillState | null)?.prefill;
  const validPrefill = prefill?.page === "benchmarks" ? prefill : undefined;
  const [tab, setTab] = useState<BenchTab>("evaluate");

  // Clear navigation state after reading to prevent re-apply on refresh
  if (prefill) {
    window.history.replaceState({}, "");
  }

  return (
    <>
      <PageHeader title="Benchmarks" />
      <TabBar tabs={TABS} active={tab} onChange={setTab} format={(t) => TAB_LABELS[t]} />
      <div className="page-body">
        {tab === "evaluate" && <EvalResultsView prefill={validPrefill} />}
        {tab === "compare" && <EvalCompareView />}
      </div>
    </>
  );
}
