import { useEffect, useRef } from "react";
import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { DOCS_CONTENT } from "./docsContent";

/** Minimal markdown-to-HTML renderer for static documentation. */
function renderMarkdown(md: string): string {
  let html = md;

  // Fenced code blocks — replace with placeholder to protect from further processing
  const codeBlocks: string[] = [];
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const idx = codeBlocks.length;
    codeBlocks.push(`<pre class="console"><code>${escaped.trimEnd()}</code></pre>`);
    return `\n%%CODEBLOCK_${idx}%%\n`;
  });

  // Tables — replace with placeholder
  const tableBlocks: string[] = [];
  html = html.replace(
    /^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)*)/gm,
    (_match, headerRow: string, bodyRows: string) => {
      const headers = headerRow.split("|").map((c: string) => c.trim()).filter(Boolean);
      const headerHtml = headers.map((h: string) => `<th>${h}</th>`).join("");
      const rows = bodyRows.trim().split("\n").map((row: string) => {
        const cells = row.split("|").map((c: string) => c.trim()).filter(Boolean);
        return `<tr>${cells.map((c: string) => `<td>${c}</td>`).join("")}</tr>`;
      }).join("");
      const idx = tableBlocks.length;
      tableBlocks.push(`<div class="docs-table-wrap"><table class="docs-table"><thead><tr>${headerHtml}</tr></thead><tbody>${rows}</tbody></table></div>`);
      return `\n%%TABLE_${idx}%%\n`;
    },
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold and italic (bold first so ** isn't caught by *)
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Process line by line
  const lines = html.split("\n");
  const result: string[] = [];
  let inList = false;

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trim();

    // Placeholders — emit directly
    if (trimmed.match(/^%%CODEBLOCK_\d+%%$/)) {
      if (inList) { result.push("</ul>"); inList = false; }
      const idx = parseInt(trimmed.match(/\d+/)![0], 10);
      result.push(codeBlocks[idx]);
      continue;
    }
    if (trimmed.match(/^%%TABLE_\d+%%$/)) {
      if (inList) { result.push("</ul>"); inList = false; }
      const idx = parseInt(trimmed.match(/\d+/)![0], 10);
      result.push(tableBlocks[idx]);
      continue;
    }

    // Headings
    const h3Anchor = trimmed.match(/^### (.+?) \{#(.+?)\}$/);
    if (h3Anchor) {
      if (inList) { result.push("</ul>"); inList = false; }
      result.push(`<h3 id="${h3Anchor[2]}">${h3Anchor[1]}</h3>`);
      continue;
    }
    if (trimmed.startsWith("### ")) {
      if (inList) { result.push("</ul>"); inList = false; }
      result.push(`<h3>${trimmed.slice(4)}</h3>`);
      continue;
    }
    if (trimmed.startsWith("## ")) {
      if (inList) { result.push("</ul>"); inList = false; }
      result.push(`<h2>${trimmed.slice(3)}</h2>`);
      continue;
    }
    if (trimmed.startsWith("# ")) {
      if (inList) { result.push("</ul>"); inList = false; }
      result.push(`<h1>${trimmed.slice(2)}</h1>`);
      continue;
    }

    // Horizontal rule
    if (trimmed === "---") {
      if (inList) { result.push("</ul>"); inList = false; }
      result.push("<hr />");
      continue;
    }

    // List items
    if (trimmed.startsWith("- ")) {
      if (!inList) { result.push("<ul>"); inList = true; }
      result.push(`<li>${trimmed.slice(2)}</li>`);
      continue;
    }

    // Empty line
    if (trimmed === "") {
      if (inList) { result.push("</ul>"); inList = false; }
      continue;
    }

    // Regular paragraph
    if (inList) { result.push("</ul>"); inList = false; }
    result.push(`<p>${trimmed}</p>`);
  }

  if (inList) result.push("</ul>");

  return result.join("\n");
}

export function DocsPage() {
  const contentRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  useEffect(() => {
    if (location.hash && contentRef.current) {
      const id = location.hash.slice(1);
      const el = contentRef.current.querySelector(`#${CSS.escape(id)}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  }, [location.hash]);

  return (
    <>
      <PageHeader title="Documentation" />
      <div
        ref={contentRef}
        className="docs-content"
        dangerouslySetInnerHTML={{ __html: renderMarkdown(DOCS_CONTENT) }}
      />
    </>
  );
}
