export type AnatomyLayerKind = "attention" | "mlp" | "moe" | "block";

export interface ModelArchitectureConfig {
  architectures?: string[];
  model_type?: string;
  hidden_dim?: number;
  hidden_size?: number;
  n_embd?: number;
  d_model?: number;
  num_layers?: number;
  num_hidden_layers?: number;
  n_layer?: number;
  attention_heads?: number;
  num_attention_heads?: number;
  n_head?: number;
  num_experts?: number;
  torch_dtype?: string;
}

export interface ModelLayerEvidence {
  jobId: string;
  jobType: string;
  label: string;
  metric?: number;
}

export interface ModelAnatomyLayer {
  index: number;
  kind: AnatomyLayerKind;
  label: string;
  detail: string;
  evidence: ModelLayerEvidence[];
}

export interface ModelAnatomyData {
  architectureLabel: string;
  attentionHeads: number | null;
  hiddenSize: number | null;
  layerCount: number;
  layers: ModelAnatomyLayer[];
  locationLabel: string;
  modelName: string;
  parameterLabel: string;
  statusLabel: string;
}
