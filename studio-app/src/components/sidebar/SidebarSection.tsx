import { ReactNode, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

const STORAGE_PREFIX = "crucible_sidebar_section_";

interface SidebarSectionProps {
  label: string;
  children: ReactNode;
}

function getInitialOpen(label: string): boolean {
  const stored = localStorage.getItem(STORAGE_PREFIX + label);
  return stored === null ? true : stored === "true";
}

export function SidebarSection({ label, children }: SidebarSectionProps) {
  const [open, setOpen] = useState(() => getInitialOpen(label));

  function toggle() {
    const next = !open;
    setOpen(next);
    localStorage.setItem(STORAGE_PREFIX + label, String(next));
  }

  return (
    <>
      <button className="sidebar-section-toggle" onClick={toggle}>
        <span className="sidebar-section-label-text">{label}</span>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
      {open && children}
    </>
  );
}
