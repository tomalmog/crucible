import { useState } from "react";
import { Check } from "lucide-react";
import { PageHeader } from "../../components/shared/PageHeader";
import { useCrucible } from "../../context/CrucibleContext";
import { FormField } from "../../components/shared/FormField";
import { HardwareProfileView } from "./HardwareProfileView";
import { getTheme, setTheme, getPaletteId, setPalette, type Theme } from "../../theme/themeUtils";
import { PALETTES } from "../../theme/palettes";

export function SettingsPage() {
  const { dataRoot, setDataRoot, refreshDatasets, hardwareProfile, refreshHardwareProfile } = useCrucible();
  const [theme, setThemeState] = useState<Theme>(getTheme());
  const [paletteId, setPaletteId] = useState(getPaletteId());

  function handleThemeChange(t: Theme) {
    setTheme(t);
    setThemeState(t);
  }

  function handlePaletteChange(id: string) {
    setPalette(id);
    setPaletteId(id);
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
          <FormField label="Color Palette">
            <div className="palette-grid">
              {PALETTES.map((p) => (
                <button
                  key={p.id}
                  className={`palette-swatch${paletteId === p.id ? " palette-swatch--active" : ""}`}
                  onClick={() => handlePaletteChange(p.id)}
                  title={p.name}
                >
                  <div className="palette-swatch-colors">
                    {p.preview.map((color, i) => (
                      <div key={i} style={{ backgroundColor: color }} />
                    ))}
                  </div>
                  <span className="palette-swatch-label">{p.name}</span>
                  {paletteId === p.id && (
                    <span className="palette-swatch-check"><Check size={14} /></span>
                  )}
                </button>
              ))}
            </div>
          </FormField>
        </div>

        <div className="panel">
          <h3 className="panel-title">Data Root</h3>
          <FormField label="Path to .crucible data directory">
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
