import { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  htmlFor?: string;
  required?: boolean;
  hint?: string;
  children: ReactNode;
}

export function FormField({ label, htmlFor, required, hint, children }: FormFieldProps) {
  return (
    <label htmlFor={htmlFor}>
      <span>
        {label}
        {required && <span className="form-field-required">*</span>}
        {hint && <span className="form-field-hint">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
