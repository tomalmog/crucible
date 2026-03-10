const THEME_KEY = "crucible-theme";

export type Theme = "light" | "dark";

export function getTheme(): Theme {
  return (localStorage.getItem(THEME_KEY) as Theme) || "light";
}

export function setTheme(theme: Theme): void {
  localStorage.setItem(THEME_KEY, theme);
  document.documentElement.setAttribute("data-theme", theme);
}

export function initTheme(): void {
  const theme = getTheme();
  if (theme === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  }
}
