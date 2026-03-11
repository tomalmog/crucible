import { Trash2, Upload, Download, Loader2 } from "lucide-react";

export interface RowItem {
  name: string;
  sizeBytes: number;
}

export function formatSize(bytes: number): string {
  if (bytes <= 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = bytes;
  for (const unit of units) {
    size /= 1024;
    if (size < 1024) return `${size.toFixed(1)} ${unit}`;
  }
  return `${size.toFixed(1)} PB`;
}

export function RegistryRow({ name, sizeBytes, selected, transferBusy, transferIcon, showTransfer, onSelect, onTransfer, onDelete }: {
  name: string;
  sizeBytes: number;
  selected?: boolean;
  transferBusy: boolean;
  transferIcon: "upload" | "download";
  showTransfer: boolean;
  onSelect?: () => void;
  onTransfer: () => void;
  onDelete: () => void;
}) {
  const TransferIcon = transferIcon === "upload" ? Upload : Download;
  const transferTitle = transferIcon === "upload" ? "Push to cluster" : "Pull to local";

  return (
    <div
      className={`flex-row${selected ? " active" : ""}`}
      style={{ alignItems: "center", padding: "4px 8px", gap: 8, cursor: onSelect ? "pointer" : undefined }}
      onClick={onSelect}
    >
      <span
        className="text-sm"
        style={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
      >
        {name}
      </span>
      <span className="text-xs text-tertiary" style={{ flexShrink: 0 }}>
        {formatSize(sizeBytes)}
      </span>
      <div style={{ display: "flex", gap: 2, flexShrink: 0 }}>
        {showTransfer && (
          <button
            className="btn btn-ghost btn-sm btn-icon"
            onClick={(e) => { e.stopPropagation(); onTransfer(); }}
            title={transferTitle}
            disabled={transferBusy}
          >
            {transferBusy
              ? <Loader2 size={12} className="spin" />
              : <TransferIcon size={12} />}
          </button>
        )}
        <button
          className="btn btn-ghost btn-sm btn-icon"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          title="Delete"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );
}
