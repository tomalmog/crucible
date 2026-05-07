import { useEffect, useState } from "react";
import { Outlet } from "react-router";
import { CrucibleProvider } from "./context/CrucibleContext";
import { CommandProvider } from "./context/CommandContext";
import { AgentChatProvider } from "./context/AgentChatContext";
import { AppSidebar } from "./components/sidebar/AppSidebar";
import { AgentSidebar } from "./components/sidebar/AgentSidebar";
import "./theme/variables.css";
import "./theme/reset.css";
import "./theme/components.css";
import "./theme/interp/page.css";
import "./theme/interp/anatomy-board.css";
import "./theme/interp/anatomy-inspector.css";
import "./theme/interp/workflow.css";
import "./theme/interp/result-shell.css";
import "./theme/interp/result-metrics.css";
import "./theme/interp/sae-steering.css";
import "./theme/interp/responsive.css";
import "./theme/form-lab.css";
import "./theme/form-lab-platform.css";
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

  useEffect(() => {
    // React to global sidebar events because layout controls live outside routed pages.
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
    // Persist agent visibility so reloads keep the user's current workspace shape.
    localStorage.setItem(AGENT_KEY, String(agentVisible));
  }, [agentVisible]);

  return (
    <CrucibleProvider>
      <CommandProvider>
        <AgentChatProvider>
          <main className={`app-shell${collapsed ? " sidebar-collapsed" : ""}${agentVisible ? " agent-open" : ""}`}>
            <AppSidebar />
            <div className="page-content">
              <Outlet />
            </div>
            <AgentSidebar onClose={() => setAgentVisible(false)} />
          </main>
        </AgentChatProvider>
      </CommandProvider>
    </CrucibleProvider>
  );
}

export default App;
