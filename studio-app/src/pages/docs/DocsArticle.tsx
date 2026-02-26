import { useEffect, useRef } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { DocEntry } from "./docsRegistry";

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "");
}

interface DocsArticleProps {
  entry: DocEntry;
}

export function DocsArticle({ entry }: DocsArticleProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    ref.current?.scrollTo({ top: 0 });
  }, [entry.slug]);

  return (
    <div ref={ref} className="docs-content docs-article">
      <h1>{entry.title}</h1>
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
        {entry.content}
      </Markdown>
    </div>
  );
}
