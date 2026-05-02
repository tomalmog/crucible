/**
 * Color palette definitions for the palette switcher.
 * Each palette generates CSS custom property overrides for both
 * light and dark themes, derived from a compact seed.
 */

export interface ColorPalette {
  id: string;
  name: string;
  preview: [string, string, string, string];
  vars: Record<string, string>;
  darkVars: Record<string, string>;
}

/* ---- Color math helpers ---- */

function hslToHex(h: number, s: number, l: number): string {
  const s1 = s / 100;
  const l1 = l / 100;
  const a = s1 * Math.min(l1, 1 - l1);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    const c = l1 - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * c).toString(16).padStart(2, "0");
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

function hexToHsl(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
    else if (max === g) h = ((b - r) / d + 2) / 6;
    else h = ((r - g) / d + 4) / 6;
  }
  return [Math.round(h * 360), Math.round(s * 100), Math.round(l * 100)];
}

function shiftLight(hex: string, amount: number): string {
  const [h, s, l] = hexToHsl(hex);
  return hslToHex(h, s, Math.max(0, Math.min(100, l + amount)));
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ---- Palette builder ---- */

interface PaletteSeed {
  id: string;
  name: string;
  preview: [string, string, string, string];
  accent: string;
  accentDark: string;
  accentLight: string;
  bgBase: string;
  sidebar: string;
  grayH: number;
  grayS: number;
}

function gray(h: number, s: number, l: number): string {
  const sFactor = Math.min(l / 50, (100 - l) / 50, 1);
  return hslToHex(h, Math.round(s * sFactor), l);
}

function makePalette(p: PaletteSeed): ColorPalette {
  const { grayH: h, grayS: s } = p;
  const darkBg = gray(h, s, 8);
  const darkSidebar = gray(h, s, 5);
  return {
    id: p.id,
    name: p.name,
    preview: p.preview,
    vars: {
      "--gray-50":  gray(h, s, 97),
      "--gray-100": gray(h, s, 95),
      "--gray-150": gray(h, s, 92),
      "--gray-200": gray(h, s, 88),
      "--gray-300": gray(h, s, 80),
      "--gray-400": gray(h, s, 64),
      "--gray-500": gray(h, s, 49),
      "--gray-600": gray(h, s, 35),
      "--gray-700": gray(h, s, 27),
      "--gray-800": gray(h, s, 18),
      "--gray-900": gray(h, s, 11),
      "--gray-950": gray(h, s, 6),
      "--accent-hue": p.accent,
      "--accent-light": p.accentLight,
      "--accent-dark": p.accentDark,
      "--bg-base": p.bgBase,
      "--bg-input": p.bgBase,
      "--sidebar-bg": p.sidebar,
      "--sidebar-bg-hover": shiftLight(p.sidebar, 6),
      "--sidebar-bg-active": shiftLight(p.sidebar, 12),
      "--sidebar-text": shiftLight(p.sidebar, 38),
      "--sidebar-text-active": p.bgBase,
      "--sidebar-text-dim": shiftLight(p.sidebar, 22),
      "--sidebar-border": "rgba(255,255,255,0.08)",
      "--text-inverse": p.bgBase,
      "--accent-muted": hexToRgba(p.accent, 0.08),
      "--chart-2": p.accentDark,
      "--shadow-sm": "0 1px 2px rgba(0,0,0,0.05)",
      "--shadow-md": "0 2px 8px rgba(0,0,0,0.08)",
      "--shadow-lg": "0 8px 24px rgba(0,0,0,0.12)",
    },
    darkVars: {
      "--gray-50":  gray(h, s, 8),
      "--gray-100": gray(h, s, 11),
      "--gray-150": gray(h, s, 14),
      "--gray-200": gray(h, s, 18),
      "--gray-300": gray(h, s, 24),
      "--gray-400": gray(h, s, 35),
      "--gray-500": gray(h, s, 49),
      "--gray-600": gray(h, s, 64),
      "--gray-700": gray(h, s, 72),
      "--gray-800": gray(h, s, 82),
      "--gray-900": gray(h, s, 90),
      "--gray-950": gray(h, s, 95),
      "--accent-hue": p.accentLight,
      "--accent-light": p.accent,
      "--accent-dark": p.accentLight,
      "--bg-base": darkBg,
      "--bg-input": gray(h, s, 11),
      "--sidebar-bg": darkSidebar,
      "--sidebar-bg-hover": shiftLight(darkSidebar, 5),
      "--sidebar-bg-active": shiftLight(darkSidebar, 10),
      "--sidebar-text": gray(h, s, 60),
      "--sidebar-text-active": gray(h, s, 92),
      "--sidebar-text-dim": gray(h, s, 35),
      "--sidebar-border": "rgba(255,255,255,0.06)",
      "--text-inverse": darkBg,
      "--accent-muted": hexToRgba(p.accentLight, 0.15),
      "--chart-2": p.accentLight,
      "--shadow-sm": "0 1px 2px rgba(0,0,0,0.3)",
      "--shadow-md": "0 2px 8px rgba(0,0,0,0.4)",
      "--shadow-lg": "0 8px 24px rgba(0,0,0,0.5)",
    },
  };
}

/* ---- Palette definitions ---- */

export const PALETTES: ColorPalette[] = [
  makePalette({
    id: "steel-teal-dark",
    name: "Steel Teal Dark",
    preview: ["#6D8196", "#B0C4DE", "#01796F", "#1e2a30"],
    accent: "#01796F", accentDark: "#015a54", accentLight: "#2a9a90",
    bgBase: "#eef2f6", sidebar: "#1e2a30",
    grayH: 210, grayS: 8,
  }),

  makePalette({
    id: "tangerine-graphite",
    name: "Tangerine Graphite",
    preview: ["#FF8C00", "#333333", "#f2f1f0", "#e07000"],
    accent: "#FF8C00", accentDark: "#e07000", accentLight: "#FFA726",
    bgBase: "#f2f1f0", sidebar: "#1f1f1f",
    grayH: 0, grayS: 0,
  }),

  makePalette({
    id: "turquoise-charcoal",
    name: "Turquoise Charcoal",
    preview: ["#00BCD4", "#363636", "#f0f2f3", "#00838F"],
    accent: "#00BCD4", accentDark: "#00838F", accentLight: "#4DD0E1",
    bgBase: "#f0f2f3", sidebar: "#2a2a2a",
    grayH: 190, grayS: 4,
  }),
];

export const DEFAULT_PALETTE_ID = "steel-teal-dark";
