import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { SidebarNavItem } from "./SidebarNavItem";
import {
  Zap, Database, Box, MessageSquare, FlaskConical, Globe, Microscope,
  Activity, Server, GitCompare, BookOpen, Settings, HardDrive, Palette,
  PanelLeftClose, PanelLeftOpen, ChevronLeft, ChevronRight, PackageOpen,
} from "lucide-react";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";

function getInitialCollapsed(): boolean {
  return localStorage.getItem(SIDEBAR_KEY) === "true";
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);
  const navigate = useNavigate();
  const location = useLocation();

  // Track navigation history to know if back/forward are available.
  // `index` points to the current position in `stack`.
  const histRef = useRef({ stack: [location.pathname], index: 0, skipNext: false });

  useEffect(() => {
    const h = histRef.current;
    if (h.skipNext) {
      h.skipNext = false;
      return;
    }
    // Normal navigation — truncate forward entries and push
    const newStack = h.stack.slice(0, h.index + 1);
    newStack.push(location.pathname);
    h.stack = newStack;
    h.index = newStack.length - 1;
  }, [location.pathname]);

  const canGoBack = histRef.current.index > 0;
  const canGoForward = histRef.current.index < histRef.current.stack.length - 1;

  function goBack() {
    if (!canGoBack) return;
    histRef.current.index -= 1;
    histRef.current.skipNext = true;
    navigate(histRef.current.stack[histRef.current.index]);
  }

  function goForward() {
    if (!canGoForward) return;
    histRef.current.index += 1;
    histRef.current.skipNext = true;
    navigate(histRef.current.stack[histRef.current.index]);
  }

  // Sync Cmd+Left / Cmd+Right (and Cmd+[ / Cmd+]) with the sidebar arrows
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (!e.metaKey && !e.ctrlKey) return;
      const back = e.key === "ArrowLeft" || e.key === "[";
      const forward = e.key === "ArrowRight" || e.key === "]";
      if (!back && !forward) return;
      e.preventDefault();
      if (back) goBack();
      else goForward();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  function toggleCollapsed() {
    const next = !collapsed;
    setCollapsed(next);
    localStorage.setItem(SIDEBAR_KEY, String(next));
    window.dispatchEvent(new CustomEvent("sidebar-toggle", { detail: next }));
  }

  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <h2>
          <span className="brand-icon">C</span>
          <span>Crucible</span>
        </h2>
        <div className="sidebar-nav-arrows">
          <button
            className={`sidebar-arrow${canGoBack ? "" : " disabled"}`}
            onClick={goBack}
            disabled={!canGoBack}
            title="Go back"
          >
            <ChevronLeft size={14} />
          </button>
          <button
            className={`sidebar-arrow${canGoForward ? "" : " disabled"}`}
            onClick={goForward}
            disabled={!canGoForward}
            title="Go forward"
          >
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      <nav className="sidebar-nav">
        <span className="sidebar-section-label">Workspace</span>
        <SidebarNavItem to="/training" icon={<Zap size={16} />} label="Training" />
        <SidebarNavItem to="/datasets" icon={<Database size={16} />} label="Datasets" />
        <SidebarNavItem to="/models" icon={<Box size={16} />} label="Models" />
        <SidebarNavItem to="/benchmarks" icon={<FlaskConical size={16} />} label="Benchmarks" />
        <SidebarNavItem to="/interpretability" icon={<Microscope size={16} />} label="Interpretability" />

        <span className="sidebar-section-label">Tools</span>
        <SidebarNavItem to="/chat" icon={<MessageSquare size={16} />} label="Chat" />
        <SidebarNavItem to="/compare-chat" icon={<GitCompare size={16} />} label="A/B Compare" />
        <SidebarNavItem to="/hub" icon={<Globe size={16} />} label="Hub" />
        <SidebarNavItem to="/export" icon={<PackageOpen size={16} />} label="Export" />

        <span className="sidebar-section-label">Operations</span>
        <SidebarNavItem to="/jobs" icon={<Activity size={16} />} label="Jobs" />
        <SidebarNavItem to="/clusters" icon={<Server size={16} />} label="Clusters" />
        <SidebarNavItem to="/resources" icon={<HardDrive size={16} />} label="Resources" />

        <div className="spacer" />

        <SidebarNavItem to="/ui-test" icon={<Palette size={16} />} label="UI Library" />
        <SidebarNavItem to="/docs" icon={<BookOpen size={16} />} label="Docs" />
        <SidebarNavItem to="/settings" icon={<Settings size={16} />} label="Settings" />
      </nav>

      <div className="sidebar-footer">
        <button
          className="sidebar-collapse-btn"
          onClick={toggleCollapsed}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        </button>
      </div>
    </aside>
  );
}
