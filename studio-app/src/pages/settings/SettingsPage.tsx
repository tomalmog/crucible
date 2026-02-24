import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { HardwareProfileView } from "./HardwareProfileView";

export function SettingsPage() {
  const { dataRoot, setDataRoot, refreshDatasets, hardwareProfile, refreshHardwareProfile } = useForge();

  return (
    <>
      <PageHeader title="Settings" />

      <div className="stack-xl">
        <div className="panel">
          <h3 className="panel-title">Data Root</h3>
          <FormField label="Path to .forge data directory">
            <input value={dataRoot} onChange={(e) => setDataRoot(e.currentTarget.value)} />
          </FormField>
          <button className="btn gap-top-sm" onClick={() => refreshDatasets().catch(console.error)}>
            Refresh Datasets
          </button>
        </div>

        <HardwareProfileView
          hardwareProfile={hardwareProfile}
          onRefresh={() => refreshHardwareProfile().catch(console.error)}
        />
      </div>
    </>
  );
}
