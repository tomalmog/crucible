import { useEffect, useMemo, useState } from "react";
import { Outlet } from "react-router";
import { CrucibleProvider } from "./context/CrucibleContext";
import { CommandProvider } from "./context/CommandContext";
import { ScriptContext, ScriptRegistration, ScriptContextValue } from "./context/ScriptContext";
import { AppSidebar } from "./components/sidebar/AppSidebar";
import { AgentSidebar } from "./components/sidebar/AgentSidebar";
import { Bot } from "lucide-react";
import "./theme/variables.css";
import "./theme/reset.css";
import "./theme/components.css";
import "./theme/layout.css";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";
const AGENT_KEY = "crucible_agent_visible";

function App() {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === "true"
  );
  const [agentVisible, setAgentVisible] = useState(
    () => localStorage.getItem(AGENT_KEY) === "true"
  );
  const [scriptReg, setScriptReg] = useState<ScriptRegistration | null>(null);
  const scriptCtx = useMemo<ScriptContextValue>(() => ({
    registration: scriptReg,
    register: (reg: ScriptRegistration) => setScriptReg(reg),
    unregister: () => setScriptReg(null),
  }), [scriptReg]);

  useEffect(() => {
    function onToggle(e: Event) {
      setCollapsed((e as CustomEvent).detail as boolean);
    }
    function onAgentToggle() {
      setAgentVisible((prev) => {
        const next = !prev;
        localStorage.setItem(AGENT_KEY, String(next));
        return next;
      });
    }
    window.addEventListener("sidebar-toggle", onToggle);
    window.addEventListener("agent-toggle", onAgentToggle);
    return () => {
      window.removeEventListener("sidebar-toggle", onToggle);
      window.removeEventListener("agent-toggle", onAgentToggle);
    };
  }, []);

  useEffect(() => {
    localStorage.setItem(AGENT_KEY, String(agentVisible));
  }, [agentVisible]);

  return (
    <CrucibleProvider>
      <CommandProvider>
        <ScriptContext.Provider value={scriptCtx}>
        <main className={`app-shell${collapsed ? " sidebar-collapsed" : ""}${agentVisible ? " agent-open" : ""}`}>
          <AppSidebar />
          <div className="page-content">
            <Outlet />
            {!agentVisible && (
              <button
                className="agent-toggle-fab"
                onClick={() => setAgentVisible(true)}
                title="Open AI Agent"
              >
                <Bot size={18} />
              </button>
            )}
          </div>
          {agentVisible && (
            <AgentSidebar onClose={() => setAgentVisible(false)} />
          )}
        </main>
        </ScriptContext.Provider>
      </CommandProvider>
    </CrucibleProvider>
  );
}

export default App;
