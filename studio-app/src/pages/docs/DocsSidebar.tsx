import { DOC_ENTRIES, CATEGORY_ORDER } from "./docsRegistry";
import type { DocCategory } from "./docsRegistry";

interface DocsSidebarProps {
  activeSlug: string;
  onSelect: (slug: string) => void;
}

export function DocsSidebar({ activeSlug, onSelect }: DocsSidebarProps) {
  const grouped = new Map<DocCategory, typeof DOC_ENTRIES>();
  for (const cat of CATEGORY_ORDER) grouped.set(cat, []);
  for (const entry of DOC_ENTRIES) {
    grouped.get(entry.category)?.push(entry);
  }

  return (
    <nav className="docs-sidebar">
      {CATEGORY_ORDER.map((cat) => {
        const entries = grouped.get(cat);
        if (!entries?.length) return null;
        return (
          <div key={cat} className="docs-sidebar-group">
            <div className="docs-sidebar-category">{cat}</div>
            {entries.map((entry) => (
              <button
                key={entry.slug}
                className={`docs-sidebar-item${entry.slug === activeSlug ? " active" : ""}`}
                onClick={() => onSelect(entry.slug)}
              >
                {entry.title}
              </button>
            ))}
          </div>
        );
      })}
    </nav>
  );
}
