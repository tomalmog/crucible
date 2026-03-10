import { Outlet } from "react-router";
import { CrucibleProvider } from "./context/CrucibleContext";
import { CommandProvider } from "./context/CommandContext";
import { AppSidebar } from "./components/sidebar/AppSidebar";
import "./theme/variables.css";
import "./theme/reset.css";
import "./theme/components.css";
import "./theme/layout.css";

function App() {
  return (
    <CrucibleProvider>
      <CommandProvider>
        <main className="app-shell">
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
