import { useEffect, useState } from "react";
import { Outlet } from "react-router";
import { CrucibleProvider } from "./context/CrucibleContext";
import { CommandProvider } from "./context/CommandContext";
import { AppSidebar } from "./components/sidebar/AppSidebar";
import "./theme/variables.css";
import "./theme/reset.css";
import "./theme/components.css";
import "./theme/layout.css";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";

function App() {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_KEY) === "true"
  );

  useEffect(() => {
    function onToggle(e: Event) {
      setCollapsed((e as CustomEvent).detail as boolean);
    }
    window.addEventListener("sidebar-toggle", onToggle);
    return () => window.removeEventListener("sidebar-toggle", onToggle);
  }, []);

  return (
    <CrucibleProvider>
      <CommandProvider>
        <main className={`app-shell${collapsed ? " sidebar-collapsed" : ""}`}>
          <AppSidebar />
          <div className="page-content">
            <Outlet />
          </div>
        </main>
      </CommandProvider>
    </CrucibleProvider>
  );
}

export default App;
