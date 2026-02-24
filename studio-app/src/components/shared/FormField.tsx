import { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  htmlFor?: string;
  children: ReactNode;
}

export function FormField({ label, htmlFor, children }: FormFieldProps) {
  return (
    <label htmlFor={htmlFor}>
      {label}
      {children}
    </label>
  );
}
