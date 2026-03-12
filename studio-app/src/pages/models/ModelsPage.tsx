import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { DetailPage } from "../../components/shared/DetailPage";
import { ListRow } from "../../components/shared/ListRow";
import { EmptyState } from "../../components/shared/EmptyState";
import { RegisterModelModal } from "../../components/shared/RegisterModelModal";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { formatSize } from "../../components/shared/RegistryRow";
import { useCrucible } from "../../context/CrucibleContext";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { ModelOverview } from "./ModelOverview";
import { ModelMergeForm } from "./ModelMergeForm";
import type { ModelEntry } from "../../types/models";

type DetailTab = "overview" | "merge";
type LocationTab = "local" | "remote";
const DETAIL_TABS = ["overview", "merge"] as const;
const LOCATION_TABS = ["local", "remote"] as const;

export function ModelsPage() {
  const { dataRoot, models, setSelectedModel, refreshModels } = useCrucible();
  const command = useCrucibleCommand();
  const [detailEntry, setDetailEntry] = useState<ModelEntry | null>(null);
  const [tab, setTab] = useState<DetailTab>("overview");
  const [locationTab, setLocationTab] = useState<LocationTab>("local");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ModelEntry | null>(null);

  function handleSelect(entry: ModelEntry) {
    setSelectedModel(entry);
    setDetailEntry(entry);
    setTab("overview");
  }

  function handleBack() {
    setDetailEntry(null);
  }

  async function handleDelete(): Promise<void> {
    if (!pendingDelete) return;
    const args = ["model", "delete", "--name", pendingDelete.modelName, "--yes"];
    if (pendingDelete.hasLocal) args.push("--delete-local");
    if (pendingDelete.hasRemote) args.push("--include-remote");
    await command.run(dataRoot, args);
    setPendingDelete(null);
    if (detailEntry?.modelName === pendingDelete.modelName) setDetailEntry(null);
    await refreshModels();
  }

  async function handleRefresh(): Promise<void> {
    setIsRefreshing(true);
    await refreshModels();
    setIsRefreshing(false);
  }

  if (detailEntry) {
    return (
      <DetailPage title={detailEntry.modelName} onBack={handleBack}>
        <TabBar tabs={DETAIL_TABS} active={tab} onChange={setTab} />
        {tab === "overview" && <ModelOverview entry={detailEntry} />}
        {tab === "merge" && <ModelMergeForm />}
      </DetailPage>
    );
  }

  const filtered = models.filter((m) =>
    locationTab === "local" ? m.hasLocal : m.hasRemote,
  );

  const emptyMsg = locationTab === "local"
    ? "No local models. Train a model or download one from the Hub."
    : "No remote models.";

  return (
    <>
      <PageHeader title="Models">
        <button
          className="btn"
          onClick={() => handleRefresh().catch(console.error)}
          disabled={isRefreshing}
        >
          {isRefreshing ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      <TabBar
        tabs={LOCATION_TABS}
        active={locationTab}
        onChange={setLocationTab}
        format={(t) => t.charAt(0).toUpperCase() + t.slice(1)}
      />

      {filtered.length === 0 ? (
        <EmptyState title="No models" description={emptyMsg} />
      ) : (
        <div className="panel panel-flush">
          {filtered.map((m) => (
            <ListRow
              key={m.modelName + m.modelPath}
              name={m.modelName}
              meta={
                <>
                  {locationTab === "local" && m.hasRemote && <span className="badge">Also Remote</span>}
                  {locationTab === "remote" && m.remoteHost && (
                    <span className="badge">{m.remoteHost.split(".")[0]}</span>
                  )}
                  <span>{formatSize(m.sizeBytes)}</span>
                </>
              }
              actions={
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  title="Delete model"
                  onClick={() => setPendingDelete(m)}
                >
                  <Trash2 size={14} />
                </button>
              }
              onClick={() => handleSelect(m)}
            />
          ))}
        </div>
      )}

      <button className="fab-add" onClick={() => setShowRegister(true)} title="Register model">
        <Plus size={22} />
      </button>

      {showRegister && (
        <RegisterModelModal
          onComplete={() => setShowRegister(false)}
          onClose={() => setShowRegister(false)}
        />
      )}

      {pendingDelete && (
        <ConfirmDeleteModal
          title="Delete Model"
          itemName={pendingDelete.modelName}
          isDeleting={command.isRunning}
          onConfirm={() => handleDelete().catch(console.error)}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </>
  );
}
