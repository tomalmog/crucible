import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { EmptyState } from "../../components/shared/EmptyState";
import { ModelVersionList } from "./ModelVersionList";
import { ModelDiffView } from "./ModelDiffView";
import { ModelActions } from "./ModelActions";

type Tab = "list" | "diff" | "actions";

export function ModelsPage() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [tab, setTab] = useState<Tab>("list");
  const [listOutput, setListOutput] = useState("");

  async function loadModels() {
    const status = await command.run(dataRoot, ["model", "list"]);
    setListOutput(status.stdout);
  }

  return (
    <>
      <PageHeader title="Model Registry">
        <button className="btn btn-primary" onClick={() => loadModels().catch(console.error)} disabled={command.isRunning}>
          {command.isRunning ? "Loading..." : "Load Models"}
        </button>
      </PageHeader>

      <div className="tab-list">
        {(["list", "diff", "actions"] as Tab[]).map((t) => (
          <button key={t} className={`tab-item ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === "list" && (
        listOutput ? <ModelVersionList output={listOutput} /> : <EmptyState title="No models loaded" description="Click 'Load Models' to fetch the model registry." />
      )}
      {tab === "diff" && <ModelDiffView dataRoot={dataRoot} />}
      {tab === "actions" && <ModelActions dataRoot={dataRoot} />}

      {command.error && <p className="error-text gap-top-sm">{command.error}</p>}
    </>
  );
}
