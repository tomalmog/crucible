import { NavLink } from "react-router";
import { SidebarNavItem } from "./SidebarNavItem";
import { Zap, Database, Box, MessageSquare, Shield, Rocket, Settings } from "lucide-react";

const NAV_ITEMS = [
  { to: "/training", icon: <Zap size={16} />, label: "Training" },
  { to: "/datasets", icon: <Database size={16} />, label: "Datasets" },
  { to: "/models", icon: <Box size={16} />, label: "Models" },
  { to: "/chat", icon: <MessageSquare size={16} />, label: "Chat" },
];

const TOOLS_ITEMS = [
  { to: "/safety", icon: <Shield size={16} />, label: "Safety" },
  { to: "/deploy", icon: <Rocket size={16} />, label: "Deploy" },
  { to: "/settings", icon: <Settings size={16} />, label: "Settings" },
];

export function AppSidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <h2>
          <span className="brand-icon">F</span>
          Forge Studio
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
