export type AgentTraceEventType =
  | "status"
  | "assistant_note"
  | "tool_call"
  | "tool_result"
  | "navigation"
  | "pending_chain";

export interface AgentTraceEvent {
  type: AgentTraceEventType;
  text?: string;
  tool_name?: string;
  input_summary?: string;
  output_summary?: string;
  route?: string;
  job_id?: string;
  steps?: string[];
}

export interface AgentTrainingJobPreview {
  kind: "training";
  jobId: string;
  title: string;
  jobType: string;
  cluster: string | null;
  history: import("../types").TrainingHistory;
  finalTrainLoss: number | null;
  finalValidationLoss: number | null;
  modelPath: string | null;
}

export interface AgentEvalJobPreview {
  kind: "eval";
  jobId: string;
  title: string;
  cluster: string | null;
  averageScore: number;
  benchmarkCount: number;
  topBenchmarks: { name: string; score: number }[];
  benchmarks: AgentEvalBenchmarkScore[];
}

export interface AgentInterpJobPreview {
  kind: "interp";
  jobId: string;
  title: string;
  cluster: string | null;
  jobType: string;
  summaryLines: string[];
  result: Record<string, unknown>;
}

export type AgentJobPreview =
  | AgentTrainingJobPreview
  | AgentEvalJobPreview
  | AgentInterpJobPreview;

export interface AgentEvalBenchmarkScore {
  name: string;
  score: number;
  correct: number;
  numExamples: number;
}

export type AgentWorkspaceMode = "auto" | "focus" | "compare" | "board" | "plan";

export type AgentWorkspaceCardSelector =
  | "artifact"
  | "context"
  | "latest"
  | "latest_training"
  | "latest_eval"
  | "previous_eval"
  | "latest_interp"
  | "live_trace"
  | "pending_chain"
  | "trace";

export interface AgentWorkspaceDirective {
  mode: AgentWorkspaceMode;
  cards: AgentWorkspaceCardSelector[];
}

export interface AgentMessage {
  role: "user" | "assistant";
  content: string;
  toolsUsed?: string[];
  navigatedTo?: string;
  trace?: AgentTraceEvent[];
  artifact?: AgentJobPreview;
  workspaceDirective?: AgentWorkspaceDirective;
}

export interface AgentChatSummary {
  id: string;
  title: string;
  preview: string;
  createdAt: string;
  updatedAt: string;
  messageCount: number;
}
