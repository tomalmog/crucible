import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "react-router";
import { router } from "./router";
import { initTheme } from "./theme/themeUtils";

initTheme();

// Disable autocorrect/autocapitalize/spellcheck on ALL inputs and textareas.
// WebKit (Tauri) does not inherit these from <body>, so we stamp them directly.
function disableAutoCorrect(el: Element) {
  if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") {
    el.setAttribute("autocorrect", "off");
    el.setAttribute("autocapitalize", "off");
    el.setAttribute("spellcheck", "false");
    el.setAttribute("autocomplete", "off");
  }
}

document.querySelectorAll("input, textarea").forEach(disableAutoCorrect);
new MutationObserver((mutations) => {
  for (const m of mutations) {
    for (const node of m.addedNodes) {
      if (node.nodeType !== 1) continue;
      const el = node as Element;
      disableAutoCorrect(el);
      el.querySelectorAll?.("input, textarea").forEach(disableAutoCorrect);
    }
  }
}).observe(document.body, { childList: true, subtree: true });

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
