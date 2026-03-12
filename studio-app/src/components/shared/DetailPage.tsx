import { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";

interface DetailPageProps {
  title: string;
  onBack: () => void;
  actions?: ReactNode;
  children: ReactNode;
}

export function DetailPage({ title, onBack, actions, children }: DetailPageProps) {
  return (
    <>
      <button className="detail-back" onClick={onBack}>
        <ArrowLeft size={14} /> Back
      </button>
      <div className="page-header">
        <h1>{title}</h1>
        {actions && <div className="page-header-actions">{actions}</div>}
      </div>
      {children}
    </>
  );
}
