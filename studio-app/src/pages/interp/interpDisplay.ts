import type { LogitLensTopToken } from "../../types/interp";

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

/** Clean common tokenizer artifacts without hiding the actual decoded token. */
export function cleanToken(token: string): string {
  return token.replace(/Ġ/g, " ").replace(/Ċ/g, "\\n").replace(/â\u0096/g, "—");
}

export function describeTopToken(token?: LogitLensTopToken): string {
  if (!token) return "n/a";
  return `${cleanToken(token.token)} · ${formatPercent(token.prob, 0)}`;
}
