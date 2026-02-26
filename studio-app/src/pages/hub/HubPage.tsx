import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { HubModelSearch } from "./HubModelSearch";
import { HubDatasetSearch } from "./HubDatasetSearch";
import { HubPushForm } from "./HubPushForm";

type Tab = "models" | "datasets" | "push";

export function HubPage() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <>
      <PageHeader title="HuggingFace Hub" />

      <div className="tab-list">
        {(["models", "datasets", "push"] as Tab[]).map((t) => (
          <button
            key={t}
            className={`tab-item ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === "models" && <HubModelSearch />}
      {tab === "datasets" && <HubDatasetSearch />}
      {tab === "push" && <HubPushForm />}
    </>
  );
}
