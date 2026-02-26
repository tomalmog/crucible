import { useCallback } from "react";
import { useSearchParams } from "react-router";
import { PageHeader } from "../../components/shared/PageHeader";
import { DOC_ENTRIES } from "./docsRegistry";
import { DocsSidebar } from "./DocsSidebar";
import { DocsArticle } from "./DocsArticle";

export function DocsPage() {
  const [params, setParams] = useSearchParams();
  const activeSlug = params.get("doc") ?? DOC_ENTRIES[0].slug;
  const entry = DOC_ENTRIES.find((e) => e.slug === activeSlug) ?? DOC_ENTRIES[0];

  const onSelect = useCallback(
    (slug: string) => setParams({ doc: slug }),
    [setParams],
  );

  return (
    <>
      <PageHeader title="Documentation" />
      <div className="docs-layout">
        <DocsSidebar activeSlug={entry.slug} onSelect={onSelect} />
        <DocsArticle entry={entry} />
      </div>
    </>
  );
}
