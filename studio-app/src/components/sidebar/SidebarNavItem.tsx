import { ReactNode } from "react";
import { NavLink } from "react-router";

interface SidebarNavItemProps {
  to: string;
  icon: ReactNode;
  label: string;
}

export function SidebarNavItem({ to, icon, label }: SidebarNavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
    >
      <span className="nav-item-icon">{icon}</span>
      {label}
    </NavLink>
  );
}
