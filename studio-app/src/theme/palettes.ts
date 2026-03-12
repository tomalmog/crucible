/**
 * Color palette definitions for the palette switcher.
 * Each palette generates a full set of CSS custom property overrides.
 */

export interface ColorPalette {
  id: string;
  name: string;
  preview: [string, string, string, string];
  vars: Record<string, string>;
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
  // Reduce saturation for darker/lighter extremes
  const sFactor = Math.min(l / 50, (100 - l) / 50, 1);
  return hslToHex(h, Math.round(s * sFactor), l);
}

function makePalette(p: PaletteSeed): ColorPalette {
  const { grayH: h, grayS: s } = p;
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
  };
}

/* ---- Palette definitions ---- */

export const PALETTES: ColorPalette[] = [
  // 0 — Default (warm brown) — matches variables.css :root
  makePalette({
    id: "warm-brown",
    name: "Warm Brown",
    preview: ["#713600", "#C05800", "#FDFBD4", "#38240D"],
    accent: "#C05800", accentDark: "#713600", accentLight: "#d46a15",
    bgBase: "#fffef5", sidebar: "#38240D",
    grayH: 38, grayS: 30,
  }),

  // 1 — Red Neutral (the original Databricks-inspired)
  makePalette({
    id: "red-neutral",
    name: "Red Neutral",
    preview: ["#e04040", "#ef5350", "#ffffff", "#1b2a32"],
    accent: "#e04040", accentDark: "#c62828", accentLight: "#ef5350",
    bgBase: "#ffffff", sidebar: "#1b2a32",
    grayH: 0, grayS: 0,
  }),

  // 2 — Steel Gray
  makePalette({
    id: "steel-gray",
    name: "Steel Gray",
    preview: ["#4A4A4A", "#CBCBCB", "#FFFFE3", "#6D8196"],
    accent: "#6D8196", accentDark: "#4d6578", accentLight: "#8a9daf",
    bgBase: "#FFFFE3", sidebar: "#3a3a3a",
    grayH: 210, grayS: 6,
  }),

  // 3 — Terracotta
  makePalette({
    id: "terracotta",
    name: "Terracotta",
    preview: ["#E35336", "#F5F5DC", "#F4A460", "#A0522D"],
    accent: "#E35336", accentDark: "#A0522D", accentLight: "#F4A460",
    bgBase: "#F5F5DC", sidebar: "#3d2212",
    grayH: 28, grayS: 14,
  }),

  // 4 — Lavender Olive
  makePalette({
    id: "lavender-olive",
    name: "Lavender Olive",
    preview: ["#FDFBD4", "#BDB96A", "#C1BFFF", "#CF6DFC"],
    accent: "#CF6DFC", accentDark: "#9B3DD4", accentLight: "#dda0ff",
    bgBase: "#FDFBD4", sidebar: "#3d3b1a",
    grayH: 55, grayS: 10,
  }),

  // 5 — Purple Olive
  makePalette({
    id: "purple-olive",
    name: "Purple Olive",
    preview: ["#FFFFE3", "#DBD4FF", "#808034", "#723480"],
    accent: "#723480", accentDark: "#522060", accentLight: "#9b50b0",
    bgBase: "#FFFFE3", sidebar: "#3a3b16",
    grayH: 62, grayS: 8,
  }),

  // 6 — Golden Mauve
  makePalette({
    id: "golden-mauve",
    name: "Golden Mauve",
    preview: ["#FFBF00", "#CFB53B", "#E0B0FF", "#CCC550"],
    accent: "#CFB53B", accentDark: "#8B7A25", accentLight: "#FFBF00",
    bgBase: "#fffdf0", sidebar: "#4a4010",
    grayH: 48, grayS: 10,
  }),

  // 7 — Blush Garden
  makePalette({
    id: "blush-garden",
    name: "Blush Garden",
    preview: ["#F2C7C7", "#FFFFFF", "#D5F3D8", "#FFB7C5"],
    accent: "#e8899e", accentDark: "#c4607a", accentLight: "#FFB7C5",
    bgBase: "#ffffff", sidebar: "#4a2530",
    grayH: 350, grayS: 6,
  }),

  // 8 — Hot Pink Teal
  makePalette({
    id: "hot-pink-teal",
    name: "Hot Pink Teal",
    preview: ["#FF69B4", "#069494", "#FFFFFF", "#00F0FF"],
    accent: "#FF69B4", accentDark: "#cc4490", accentLight: "#ff8ec8",
    bgBase: "#ffffff", sidebar: "#065454",
    grayH: 180, grayS: 4,
  }),

  // 9 — Forest Maroon
  makePalette({
    id: "forest-maroon",
    name: "Forest Maroon",
    preview: ["#CBCBCB", "#F2F2F2", "#174D38", "#4D1717"],
    accent: "#174D38", accentDark: "#0d3325", accentLight: "#2a6b50",
    bgBase: "#F2F2F2", sidebar: "#4D1717",
    grayH: 0, grayS: 0,
  }),

  // 10 — Dark Mauve
  makePalette({
    id: "dark-mauve",
    name: "Dark Mauve",
    preview: ["#000000", "#D1D0D0", "#988686", "#5C4E4E"],
    accent: "#988686", accentDark: "#5C4E4E", accentLight: "#b0a0a0",
    bgBase: "#D1D0D0", sidebar: "#1a1616",
    grayH: 0, grayS: 4,
  }),

  // 11 — Teal Leather
  makePalette({
    id: "teal-leather",
    name: "Teal Leather",
    preview: ["#F2F0EF", "#BBBDBC", "#245F73", "#733E24"],
    accent: "#245F73", accentDark: "#1a4555", accentLight: "#357a92",
    bgBase: "#F2F0EF", sidebar: "#733E24",
    grayH: 200, grayS: 4,
  }),

  // 12 — Mono Charcoal
  makePalette({
    id: "mono-charcoal",
    name: "Mono Charcoal",
    preview: ["#FFFFFF", "#D4D4D4", "#B3B3B3", "#2B2B2B"],
    accent: "#555555", accentDark: "#2B2B2B", accentLight: "#777777",
    bgBase: "#FFFFFF", sidebar: "#2B2B2B",
    grayH: 0, grayS: 0,
  }),

  // 13 — Warm Blush
  makePalette({
    id: "warm-blush",
    name: "Warm Blush",
    preview: ["#F7E6CA", "#E8D59E", "#D9BBB0", "#AD9C8E"],
    accent: "#AD9C8E", accentDark: "#8a7b6e", accentLight: "#D9BBB0",
    bgBase: "#F7E6CA", sidebar: "#5a4a3a",
    grayH: 30, grayS: 12,
  }),

  // 14 — Olive Sage
  makePalette({
    id: "olive-sage",
    name: "Olive Sage",
    preview: ["#FDFBD4", "#D9D7B6", "#878672", "#545333"],
    accent: "#878672", accentDark: "#545333", accentLight: "#a0a08c",
    bgBase: "#FDFBD4", sidebar: "#3a3920",
    grayH: 58, grayS: 8,
  }),

  // 15 — Amber Honey
  makePalette({
    id: "amber-honey",
    name: "Amber Honey",
    preview: ["#FFC107", "#F9E076", "#FFFDD0", "#895129"],
    accent: "#FFC107", accentDark: "#895129", accentLight: "#F9E076",
    bgBase: "#FFFDD0", sidebar: "#895129",
    grayH: 42, grayS: 12,
  }),

  // 16 — Olive Periwinkle
  makePalette({
    id: "olive-periwinkle",
    name: "Olive Periwinkle",
    preview: ["#EDE8D0", "#6E632E", "#DBD1ED", "#ABBEED"],
    accent: "#ABBEED", accentDark: "#6E632E", accentLight: "#DBD1ED",
    bgBase: "#EDE8D0", sidebar: "#3a3518",
    grayH: 50, grayS: 8,
  }),

  // 17 — Steel Teal
  makePalette({
    id: "steel-teal",
    name: "Steel Teal",
    preview: ["#6D8196", "#B0C4DE", "#01796F", "#5A5A5A"],
    accent: "#01796F", accentDark: "#015a54", accentLight: "#2a9a90",
    bgBase: "#eef2f6", sidebar: "#5A5A5A",
    grayH: 210, grayS: 8,
  }),

  // 18 — Mint Pastel
  makePalette({
    id: "mint-pastel",
    name: "Mint Pastel",
    preview: ["#AFEEEE", "#ADEBB3", "#C4C4C4", "#D3D3D3"],
    accent: "#80cccc", accentDark: "#5aaeae", accentLight: "#AFEEEE",
    bgBase: "#f5fafa", sidebar: "#4a6060",
    grayH: 170, grayS: 4,
  }),

  // 19 — Navy Steel
  makePalette({
    id: "navy-steel",
    name: "Navy Steel",
    preview: ["#6D8196", "#ADD8E6", "#FFFAFA", "#000080"],
    accent: "#000080", accentDark: "#000060", accentLight: "#3333b0",
    bgBase: "#FFFAFA", sidebar: "#1a1a4a",
    grayH: 230, grayS: 6,
  }),

  // 20 — Forest Green
  makePalette({
    id: "forest-green",
    name: "Forest Green",
    preview: ["#808000", "#228B22", "#636B2F", "#F2F0EF"],
    accent: "#228B22", accentDark: "#1a6b1a", accentLight: "#3aaf3a",
    bgBase: "#F2F0EF", sidebar: "#2d3016",
    grayH: 80, grayS: 6,
  }),

  // 21 — Sage Green
  makePalette({
    id: "sage-green",
    name: "Sage Green",
    preview: ["#B2AC88", "#898989", "#F2F0EF", "#4B6E48"],
    accent: "#4B6E48", accentDark: "#3a5538", accentLight: "#6a8e66",
    bgBase: "#F2F0EF", sidebar: "#3a4a38",
    grayH: 50, grayS: 6,
  }),
];

export const DEFAULT_PALETTE_ID = "warm-brown";
