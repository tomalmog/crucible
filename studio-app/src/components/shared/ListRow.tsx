import { ReactNode } from "react";
import { ChevronRight } from "lucide-react";

interface ListRowProps {
  name: string;
  meta?: ReactNode;
  actions?: ReactNode;
  showChevron?: boolean;
  onClick?: () => void;
}

export function ListRow({ name, meta, actions, showChevron = true, onClick }: ListRowProps) {
  return (
    <div className="list-row" onClick={onClick}>
      <span className="list-row-name">{name}</span>
      {meta && <div className="list-row-meta">{meta}</div>}
      {actions && (
        <div className="list-row-actions" onClick={(e) => e.stopPropagation()}>
          {actions}
        </div>
      )}
      {showChevron && <ChevronRight size={14} className="list-row-chevron" />}
    </div>
  );
}
