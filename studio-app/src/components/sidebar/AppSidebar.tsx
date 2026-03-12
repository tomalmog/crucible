import { NavLink } from "react-router";
import { SidebarNavItem } from "./SidebarNavItem";
import { Zap, Database, Box, MessageSquare, Settings, FlaskConical, Globe, GitCompare, Activity, BookOpen, Server } from "lucide-react";

const NAV_ITEMS = [
  { to: "/training", icon: <Zap size={16} />, label: "Training" },
  { to: "/datasets", icon: <Database size={16} />, label: "Datasets" },
  { to: "/models", icon: <Box size={16} />, label: "Models" },
  { to: "/chat", icon: <MessageSquare size={16} />, label: "Chat" },
  { to: "/benchmarks", icon: <FlaskConical size={16} />, label: "Benchmarks" },
  { to: "/hub", icon: <Globe size={16} />, label: "Hub" },
];

const TOOLS_ITEMS = [
  { to: "/jobs", icon: <Activity size={16} />, label: "Jobs" },
  { to: "/clusters", icon: <Server size={16} />, label: "Clusters" },
  { to: "/compare-chat", icon: <GitCompare size={16} />, label: "A/B Compare" },
  { to: "/docs", icon: <BookOpen size={16} />, label: "Docs" },
  { to: "/settings", icon: <Settings size={16} />, label: "Settings" },
];

export function AppSidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <h2>
          <span className="brand-icon">C</span>
          Crucible
        </h2>
      </div>
      <nav className="sidebar-nav">
        <div className="sidebar-section-label">Workspace</div>
        {NAV_ITEMS.map((item) => (
          <SidebarNavItem key={item.to} {...item} />
        ))}
        <div className="sidebar-section-label">Tools</div>
        {TOOLS_ITEMS.map((item) => (
          <SidebarNavItem key={item.to} {...item} />
        ))}
      </nav>
      <div className="sidebar-footer">
        <NavLink to="/settings" className="nav-item">
          <span className="nav-item-icon"><Settings size={16} /></span>
          Settings
        </NavLink>
      </div>
    </aside>
  );
}
