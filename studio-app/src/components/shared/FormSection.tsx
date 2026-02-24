import { ReactNode, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface FormSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function FormSection({ title, defaultOpen = false, children }: FormSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  return (
    <div className="form-section-divider">
      <button
        type="button"
        className="btn btn-ghost btn-sm form-section-toggle"
        onClick={() => setIsOpen((o) => !o)}
      >
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {title}
      </button>
      {isOpen && <div className="form-section-body">{children}</div>}
    </div>
  );
}
