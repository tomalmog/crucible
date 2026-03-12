import { PALETTES, DEFAULT_PALETTE_ID, type ColorPalette } from "./palettes";

const THEME_KEY = "crucible-theme";
const PALETTE_KEY = "crucible-palette";

export type Theme = "light" | "dark";

export function getTheme(): Theme {
  return (localStorage.getItem(THEME_KEY) as Theme) || "light";
}

export function setTheme(theme: Theme): void {
  localStorage.setItem(THEME_KEY, theme);
  document.documentElement.setAttribute("data-theme", theme);
}

/* ---- Palette ---- */

export function getPaletteId(): string {
  return localStorage.getItem(PALETTE_KEY) || DEFAULT_PALETTE_ID;
}

export function applyPalette(palette: ColorPalette): void {
  const root = document.documentElement;
  for (const [prop, value] of Object.entries(palette.vars)) {
    root.style.setProperty(prop, value);
  }
}

export function setPalette(id: string): void {
  const palette = PALETTES.find((p) => p.id === id);
  if (!palette) return;
  localStorage.setItem(PALETTE_KEY, id);
  applyPalette(palette);
}

/* ---- Init (call once on app load) ---- */

export function initTheme(): void {
  const theme = getTheme();
  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  }
  const paletteId = getPaletteId();
  const palette = PALETTES.find((p) => p.id === paletteId);
  if (palette) applyPalette(palette);
}
