import { useEffect, useRef } from "react";
import { useLocation } from "react-router";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { PageHeader } from "../../components/shared/PageHeader";
import { DOCS_CONTENT } from "./docsContent";

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "");
}

export function DocsPage() {
  const contentRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  // Scroll to anchor when URL hash changes (e.g. clicking doc section links)
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
      <div ref={contentRef} className="docs-content">
        <Markdown
          remarkPlugins={[remarkGfm]}
          components={{
            h3: ({ children }) => {
              const text = String(children);
              return <h3 id={slugify(text)}>{children}</h3>;
            },
            pre: ({ children }) => <pre className="console">{children}</pre>,
            table: ({ children }) => (
              <div className="docs-table-wrap">
                <table className="docs-table">{children}</table>
              </div>
            ),
          }}
        >
          {DOCS_CONTENT}
        </Markdown>
      </div>
    </>
  );
}
