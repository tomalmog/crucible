import { useEffect, useRef } from "react";
import { useLocation } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { DOCS_CONTENT } from "./docsContent";

/** Minimal markdown-to-HTML renderer for static documentation. */
function renderMarkdown(md: string): string {
  let html = md;

  // Fenced code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `<pre class="console"><code>${escaped.trimEnd()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Tables
  html = html.replace(
    /^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)*)/gm,
    (_match, headerRow: string, bodyRows: string) => {
      const headers = headerRow.split("|").map((c: string) => c.trim()).filter(Boolean);
      const headerHtml = headers.map((h: string) => `<th>${h}</th>`).join("");
      const rows = bodyRows.trim().split("\n").map((row: string) => {
        const cells = row.split("|").map((c: string) => c.trim()).filter(Boolean);
        return `<tr>${cells.map((c: string) => `<td>${c}</td>`).join("")}</tr>`;
      }).join("");
      return `<div class="docs-table-wrap"><table class="docs-table"><thead><tr>${headerHtml}</tr></thead><tbody>${rows}</tbody></table></div>`;
    },
  );

  // Headings with anchors: ### Heading {#anchor}
  html = html.replace(/^### (.+?) \{#(.+?)\}\s*$/gm, '<h3 id="$2">$1</h3>');
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Horizontal rules
  html = html.replace(/^---$/gm, "<hr />");

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Paragraphs: wrap lines that aren't already block elements
  const lines = html.split("\n");
  const result: string[] = [];
  let inBlock = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("<pre") || trimmed.startsWith("<div class=\"docs-table")) {
      inBlock = true;
      result.push(line);
      continue;
    }
    if (trimmed.startsWith("</pre>") || trimmed.endsWith("</table></div>")) {
      inBlock = false;
      result.push(line);
      continue;
    }
    if (inBlock) {
      result.push(line);
      continue;
    }
    if (
      trimmed === "" ||
      trimmed.startsWith("<h") ||
      trimmed.startsWith("<hr") ||
      trimmed.startsWith("<pre") ||
      trimmed.startsWith("<div")
    ) {
      result.push(line);
      continue;
    }
    result.push(`<p>${trimmed}</p>`);
  }

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
