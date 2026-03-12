import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { HubModelSearch } from "./HubModelSearch";
import { HubDatasetSearch } from "./HubDatasetSearch";
import { HubPushForm } from "./HubPushForm";

type Tab = "models" | "datasets" | "push";
const HUB_TABS = ["models", "datasets", "push"] as const;

export function HubPage() {
  const [tab, setTab] = useState<Tab>("models");

  return (
    <>
      <PageHeader title="HuggingFace Hub" />
      <TabBar tabs={HUB_TABS} active={tab} onChange={setTab} />
      {tab === "models" && <HubModelSearch />}
      {tab === "datasets" && <HubDatasetSearch />}
      {tab === "push" && <HubPushForm />}
    </>
  );
}
