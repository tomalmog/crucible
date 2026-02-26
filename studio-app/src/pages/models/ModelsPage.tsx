import { useState, useEffect, useCallback } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { useForgeCommand } from "../../hooks/useForgeCommand";
import { EmptyState } from "../../components/shared/EmptyState";
import { ModelListPanel } from "./ModelListPanel";
import { ModelOverview } from "./ModelOverview";
import { ModelDiffView } from "./ModelDiffView";
import { ModelActions } from "./ModelActions";
import { ModelMergeForm } from "./ModelMergeForm";
import { parseModelList } from "./parseModelList";
import type { ModelVersion } from "../../types/models";

type Tab = "overview" | "diff" | "actions" | "merge";

export function ModelsPage() {
  const { dataRoot } = useForge();
  const command = useForgeCommand();
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [selected, setSelected] = useState<ModelVersion | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  const loadModels = useCallback(async () => {
    const status = await command.run(dataRoot, ["model", "list"]);
    const parsed = parseModelList(status.stdout);
    setVersions(parsed);
    if (parsed.length > 0 && !selected) setSelected(parsed[0]);
  }, [dataRoot]);

  useEffect(() => { loadModels().catch(console.error); }, []);

  return (
    <>
      <PageHeader title="Model Registry">
        <button className="btn" onClick={() => loadModels().catch(console.error)} disabled={command.isRunning}>
          {command.isRunning ? "Loading..." : "Refresh"}
        </button>
      </PageHeader>

      <div className="two-column">
        <ModelListPanel versions={versions} selected={selected} onSelect={setSelected} />
        <div>
          <div className="tab-list">
            {(["overview", "diff", "actions", "merge"] as Tab[]).map((t) => (
              <button key={t} className={`tab-item ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>

          {tab === "overview" && (
            selected
              ? <ModelOverview version={selected} />
              : <EmptyState title="No model selected" description="Select a model version from the list." />
          )}
          {tab === "diff" && <ModelDiffView dataRoot={dataRoot} versions={versions} />}
          {tab === "actions" && <ModelActions dataRoot={dataRoot} versions={versions} />}
          {tab === "merge" && <ModelMergeForm />}
        </div>
      </div>

      {command.error && <p className="error-text gap-top-sm">{command.error}</p>}
    </>
  );
}
