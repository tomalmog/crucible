import { useState } from "react";
import { SidebarNavItem } from "./SidebarNavItem";
import {
  Zap, Database, Box, MessageSquare, FlaskConical, Globe,
  Activity, Server, GitCompare, BookOpen, Settings,
  PanelLeftClose, PanelLeftOpen,
} from "lucide-react";

const SIDEBAR_KEY = "crucible_sidebar_collapsed";

function getInitialCollapsed(): boolean {
  return localStorage.getItem(SIDEBAR_KEY) === "true";
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(getInitialCollapsed);

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
      </div>

      <nav className="sidebar-nav">
        <SidebarNavItem to="/training" icon={<Zap size={16} />} label="Training" />
        <SidebarNavItem to="/datasets" icon={<Database size={16} />} label="Datasets" />
        <SidebarNavItem to="/models" icon={<Box size={16} />} label="Models" />
        <SidebarNavItem to="/chat" icon={<MessageSquare size={16} />} label="Chat" />
        <SidebarNavItem to="/benchmarks" icon={<FlaskConical size={16} />} label="Benchmarks" />
        <SidebarNavItem to="/hub" icon={<Globe size={16} />} label="Hub" />

        <div className="sidebar-divider" />

        <SidebarNavItem to="/jobs" icon={<Activity size={16} />} label="Jobs" />
        <SidebarNavItem to="/clusters" icon={<Server size={16} />} label="Clusters" />
        <SidebarNavItem to="/compare-chat" icon={<GitCompare size={16} />} label="A/B Compare" />
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
