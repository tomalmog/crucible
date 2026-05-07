import type { TrainingMethod } from "../../types/training";
import { BasicTrainForm } from "./forms/BasicTrainForm";
import { DistillTrainForm } from "./forms/DistillTrainForm";
import { DomainAdaptForm } from "./forms/DomainAdaptForm";
import { DpoTrainForm } from "./forms/DpoTrainForm";
import { GrpoTrainForm } from "./forms/GrpoTrainForm";
import { KtoTrainForm } from "./forms/KtoTrainForm";
import { LoraTrainForm } from "./forms/LoraTrainForm";
import { MultimodalTrainForm } from "./forms/MultimodalTrainForm";
import { OrpoTrainForm } from "./forms/OrpoTrainForm";
import { QloraTrainForm } from "./forms/QloraTrainForm";
import { RlhfTrainForm } from "./forms/RlhfTrainForm";
import { RlvrTrainForm } from "./forms/RlvrTrainForm";
import { SftTrainForm } from "./forms/SftTrainForm";

interface TrainingMethodConfigProps {
  extra: Record<string, string>;
  method: TrainingMethod;
  setExtra: (extra: Record<string, string>) => void;
}

export function TrainingMethodConfig({
  extra,
  method,
  setExtra,
}: TrainingMethodConfigProps) {
  if (method === "train") return <BasicTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "sft") return <SftTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "dpo-train") return <DpoTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "rlhf-train") return <RlhfTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "lora-train") return <LoraTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "distill") return <DistillTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "domain-adapt") return <DomainAdaptForm extra={extra} setExtra={setExtra} />;
  if (method === "grpo-train") return <GrpoTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "qlora-train") return <QloraTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "kto-train") return <KtoTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "orpo-train") return <OrpoTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "multimodal-train") return <MultimodalTrainForm extra={extra} setExtra={setExtra} />;
  if (method === "rlvr-train") return <RlvrTrainForm extra={extra} setExtra={setExtra} />;
  return null;
}
