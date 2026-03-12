import { ReactNode, useId } from "react";

interface FormFieldProps {
  label: string;
  htmlFor?: string;
  required?: boolean;
  hint?: string;
  children: ReactNode;
}

export function FormField({ label, htmlFor, required, hint, children }: FormFieldProps) {
  const autoId = useId();
  const id = htmlFor ?? autoId;

  return (
    <div className="ff">
      <label className="ff-label" htmlFor={id}>
        {label}
        {required && <span className="ff-required">*</span>}
        {hint && <span className="ff-hint">{hint}</span>}
      </label>
      {children}
    </div>
  );
}
