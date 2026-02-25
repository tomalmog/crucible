import { useState } from "react";
import { HubModelSearch } from "./HubModelSearch";
import { HubDatasetSearch } from "./HubDatasetSearch";
import { HubPushForm } from "./HubPushForm";

type Tab = "models" | "datasets" | "push";

export function HubPage() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <div>
      <div className="page-header">
        <h1>HuggingFace Hub</h1>
      </div>
      <div className="tab-bar">
        <button className={`tab${tab === "models" ? " active" : ""}`} onClick={() => setTab("models")}>
          Models
        </button>
        <button className={`tab${tab === "datasets" ? " active" : ""}`} onClick={() => setTab("datasets")}>
          Datasets
        </button>
        <button className={`tab${tab === "push" ? " active" : ""}`} onClick={() => setTab("push")}>
          Push
        </button>
      </div>
      {tab === "models" && <HubModelSearch />}
      {tab === "datasets" && <HubDatasetSearch />}
      {tab === "push" && <HubPushForm />}
    </div>
  );
}
