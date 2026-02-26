import { open } from "@tauri-apps/plugin-dialog";
import { FolderOpen } from "lucide-react";

interface PathInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  kind?: "file" | "folder";
  filters?: { name: string; extensions: string[] }[];
}

export function PathInput({ value, onChange, placeholder, disabled, kind = "file", filters }: PathInputProps) {
  async function browse() {
    const result = await open({
      directory: kind === "folder",
      multiple: false,
      filters,
    });
    if (result) onChange(result as string);
  }

  return (
    <div className="path-input">
      <input value={value} onChange={(e) => onChange(e.currentTarget.value)} placeholder={placeholder} disabled={disabled} />
      <button type="button" className="btn btn-ghost btn-sm path-input-browse" onClick={browse} disabled={disabled} title="Browse...">
        <FolderOpen size={14} />
      </button>
    </div>
  );
}
