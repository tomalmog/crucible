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
        {required && <span style={{ color: "var(--error)", marginLeft: 2 }}>*</span>}
        {hint && (
          <span style={{
            fontWeight: 400,
            color: "var(--text-tertiary)",
            fontSize: "0.625rem",
            marginLeft: 6,
          }}>
            {hint}
          </span>
        )}
      </span>
      {children}
    </label>
  );
}
