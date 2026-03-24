/** Export result types for ONNX, SafeTensors, and GGUF. */

export interface OnnxExportResult {
  onnx_path: string;
  file_size_mb: number;
  opset_version: number;
  verification: string;
  input_names: string[];
  output_names: string[];
  tokenizer_copied: boolean;
}

export interface SafeTensorsExportResult {
  output_path: string;
  file_size_mb: number;
  num_tensors: number;
  tokenizer_copied: boolean;
}

export interface GgufExportResult {
  output_path: string;
  file_size_mb: number;
  quant_type: string;
  num_tensors: number;
  tokenizer_copied: boolean;
}

export interface HfExportResult {
  output_path: string;
  file_size_mb: number;
  num_tensors: number;
  config_generated: boolean;
  tokenizer_copied: boolean;
}
