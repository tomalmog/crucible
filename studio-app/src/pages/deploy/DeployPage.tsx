import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { PackageForm } from "./PackageForm";
import { QuantizeForm } from "./QuantizeForm";
import { LatencyProfileForm } from "./LatencyProfileForm";
import { ReadinessChecklist } from "./ReadinessChecklist";

type Tab = "package" | "quantize" | "profile" | "checklist";

export function DeployPage() {
  const { dataRoot } = useCrucible();
  const [tab, setTab] = useState<Tab>("package");

  return (
    <>
      <PageHeader title="Deploy" />

      <div className="tab-list">
        {(["package", "quantize", "profile", "checklist"] as Tab[]).map((t) => (
          <button key={t} className={`tab-item ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === "package" && <PackageForm dataRoot={dataRoot} />}
      {tab === "quantize" && <QuantizeForm dataRoot={dataRoot} />}
      {tab === "profile" && <LatencyProfileForm dataRoot={dataRoot} />}
      {tab === "checklist" && <ReadinessChecklist dataRoot={dataRoot} />}
    </>
  );
}
