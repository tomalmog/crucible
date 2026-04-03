import { useState } from "react";
import { Check, Eye, EyeOff } from "lucide-react";

const AGENT_API_KEY = "crucible_anthropic_api_key";
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
  const [agentProvider, setAgentProvider] = useState(() => localStorage.getItem("crucible_agent_provider") ?? "anthropic");
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(AGENT_API_KEY) ?? "");
  const [apiKeyVisible, setApiKeyVisible] = useState(false);
  const [ollamaModel, setOllamaModel] = useState(() => localStorage.getItem("crucible_agent_ollama_model") ?? "llama3.1");
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem("crucible_agent_ollama_url") ?? "http://localhost:11434");
  const [geminiModel, setGeminiModel] = useState(() => localStorage.getItem("crucible_agent_gemini_model") ?? "gemini-2.5-flash");
  const [geminiApiKey, setGeminiApiKey] = useState(() => localStorage.getItem("crucible_gemini_api_key") ?? "");
  const [geminiApiKeyVisible, setGeminiApiKeyVisible] = useState(false);

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

        <div className="panel">
          <h3 className="panel-title">AI Agent</h3>
          <FormField label="Provider">
            <select
              value={agentProvider}
              onChange={(e) => {
                const val = e.target.value;
                setAgentProvider(val);
                localStorage.setItem("crucible_agent_provider", val);
              }}
            >
              <option value="anthropic">Anthropic (Claude API)</option>
              <option value="ollama">Ollama (Local)</option>
              <option value="gemini">Google Gemini (Vertex AI)</option>
            </select>
          </FormField>
          {agentProvider === "anthropic" && (
            <FormField label="Anthropic API Key">
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  type={apiKeyVisible ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => {
                    const val = e.currentTarget.value;
                    setApiKey(val);
                    localStorage.setItem(AGENT_API_KEY, val);
                  }}
                  placeholder="sk-ant-..."
                  style={{ flex: 1 }}
                />
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => setApiKeyVisible(!apiKeyVisible)}
                  title={apiKeyVisible ? "Hide" : "Show"}
                >
                  {apiKeyVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <p className="ff-hint">Also checks ANTHROPIC_API_KEY env var.</p>
            </FormField>
          )}
          {agentProvider === "ollama" && (
            <>
              <FormField label="Model">
                <input
                  value={ollamaModel}
                  onChange={(e) => {
                    const val = e.currentTarget.value;
                    setOllamaModel(val);
                    localStorage.setItem("crucible_agent_ollama_model", val);
                  }}
                  placeholder="llama3.1"
                />
              </FormField>
              <FormField label="Ollama URL">
                <input
                  value={ollamaUrl}
                  onChange={(e) => {
                    const val = e.currentTarget.value;
                    setOllamaUrl(val);
                    localStorage.setItem("crucible_agent_ollama_url", val);
                  }}
                  placeholder="http://localhost:11434"
                />
              </FormField>
            </>
          )}
          {agentProvider === "gemini" && (
            <>
              <FormField label="Gemini API Key">
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    type={geminiApiKeyVisible ? "text" : "password"}
                    value={geminiApiKey}
                    onChange={(e) => {
                      const val = e.currentTarget.value;
                      setGeminiApiKey(val);
                      localStorage.setItem("crucible_gemini_api_key", val);
                    }}
                    placeholder="AIza..."
                    style={{ flex: 1 }}
                  />
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => setGeminiApiKeyVisible(!geminiApiKeyVisible)}
                    title={geminiApiKeyVisible ? "Hide" : "Show"}
                  >
                    {geminiApiKeyVisible ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
                <p className="ff-hint">
                  API key from aistudio.google.com. Also checks GOOGLE_API_KEY env var.
                  Leave blank to use Vertex AI with gcloud auth instead.
                </p>
              </FormField>
              <FormField label="Model">
                <input
                  value={geminiModel}
                  onChange={(e) => {
                    const val = e.currentTarget.value;
                    setGeminiModel(val);
                    localStorage.setItem("crucible_agent_gemini_model", val);
                  }}
                  placeholder="gemini-2.5-flash"
                />
              </FormField>
            </>
          )}
        </div>

        <HardwareProfileView
          hardwareProfile={hardwareProfile}
          onRefresh={() => refreshHardwareProfile().catch(console.error)}
        />
      </div>
    </>
  );
}
