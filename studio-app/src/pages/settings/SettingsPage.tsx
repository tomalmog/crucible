import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useForge } from "../../context/ForgeContext";
import { FormField } from "../../components/shared/FormField";
import { HardwareProfileView } from "./HardwareProfileView";
import { getTheme, setTheme, type Theme } from "../../theme/themeUtils";

export function SettingsPage() {
  const { dataRoot, setDataRoot, refreshDatasets, hardwareProfile, refreshHardwareProfile } = useForge();
  const [theme, setThemeState] = useState<Theme>(getTheme());

  function handleThemeChange(t: Theme) {
    setTheme(t);
    setThemeState(t);
  }

  return (
    <>
      <PageHeader title="Settings" />

      <div className="stack-xl">
        <div className="panel">
          <h3 className="panel-title">Appearance</h3>
          <FormField label="Theme">
            <select value={theme} onChange={(e) => handleThemeChange(e.target.value as Theme)}>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </FormField>
        </div>

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
