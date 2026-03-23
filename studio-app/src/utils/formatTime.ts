/** Format an ISO timestamp or epoch ms as a relative time string like "3m ago", "2h ago", "Mar 15". */
export function formatTimeAgo(input: string | number): string {
  const ms = typeof input === "string" ? new Date(input).getTime() : input;
  if (!ms || isNaN(ms)) return "";
  const diff = Date.now() - ms;
  if (diff < 0) return "just now";
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  const d = new Date(ms);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
