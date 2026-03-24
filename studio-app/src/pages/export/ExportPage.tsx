import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { TabBar } from "../../components/shared/TabBar";
import { OnnxExportForm } from "./OnnxExportForm";
import { SafeTensorsExportForm } from "./SafeTensorsExportForm";
import { GgufExportForm } from "./GgufExportForm";
import { HfExportForm } from "./HfExportForm";

type ExportTab = "onnx" | "safetensors" | "gguf" | "huggingface";
const TABS = ["onnx", "safetensors", "gguf", "huggingface"] as const;
const TAB_LABELS: Record<ExportTab, string> = {
  onnx: "ONNX",
  safetensors: "SafeTensors",
  gguf: "GGUF",
  huggingface: "HuggingFace",
};

export function ExportPage() {
  const [tab, setTab] = useState<ExportTab>("onnx");

  return (
    <>
      <PageHeader title="Export" />
      <TabBar tabs={TABS} active={tab} onChange={setTab} format={(t) => TAB_LABELS[t]} />
      {tab === "onnx" && <OnnxExportForm />}
      {tab === "safetensors" && <SafeTensorsExportForm />}
      {tab === "gguf" && <GgufExportForm />}
      {tab === "huggingface" && <HfExportForm />}
    </>
  );
}
