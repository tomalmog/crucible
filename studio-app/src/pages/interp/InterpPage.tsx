import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { ModelHealthOverview } from "./ModelHealthOverview";

export function InterpPage() {
  const location = useLocation();
  const prefill = readPrefill(location.state);
  const validPrefill = prefill?.page === "interpretability" ? prefill : undefined;

  // Clear navigation state after reading to prevent re-apply on refresh
  if (prefill) {
    window.history.replaceState({}, "");
  }

  return (
    <>
      <PageHeader title="Model Health" />
      <div className="page-body">
        <ModelHealthOverview advancedPrefill={validPrefill} />
      </div>
    </>
  );
}

function readPrefill(state: unknown): Record<string, unknown> | undefined {
  if (!isRecord(state) || !("prefill" in state)) return undefined;
  const prefill = state.prefill;
  return isRecord(prefill) ? prefill : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
