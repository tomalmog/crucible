import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { HubModelSearch } from "./HubModelSearch";
import { HubDatasetSearch } from "./HubDatasetSearch";

type Tab = "models" | "datasets";
const HUB_TABS = ["models", "datasets"] as const;

export function HubPage() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <>
      <PageHeader title="HuggingFace Hub" />
      <TabBar tabs={HUB_TABS} active={tab} onChange={setTab} />
      {tab === "models" && <HubModelSearch />}
      {tab === "datasets" && <HubDatasetSearch />}
    </>
  );
}
