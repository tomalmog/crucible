import type { DocEntry } from "../docsRegistry";

export const deployment: DocEntry = {
  slug: "deployment",
  title: "Deployment",
  category: "Deployment",
  content: `
## Deployment

Once your model is trained and evaluated, Forge helps you package and prepare it for production inference.

### Model Packaging

Every completed training run produces a self-contained output directory with everything needed for inference: **model weights**, **tokenizer files**, and **configuration**. This directory can be loaded directly by HuggingFace Transformers, vLLM, or any compatible serving framework without extra setup.

### ONNX Export

Convert a PyTorch model to ONNX format for cross-platform inference:

\`\`\`bash
forge export --format onnx --model <path> --output <path>
\`\`\`

ONNX models run on a wide range of hardware and runtimes including ONNX Runtime, TensorRT, and CoreML. This is the recommended path for deploying models outside of Python or on edge devices.

### Quantization

Reduce model size and speed up inference with post-training quantization. Forge supports **int8** and **int4** quantization, which can shrink model size by 2-4x with minimal quality loss. Quantized models load faster, use less memory, and run significantly quicker on both CPU and GPU.

### Hardware Profiling

Understand your deployment target before shipping:

\`\`\`bash
forge hardware-profile
\`\`\`

This detects available GPUs, measures memory capacity, estimates maximum batch sizes, and reports compute throughput. Use this information to choose the right quantization level and batch configuration for your serving environment.

### Readiness Checks

Before deploying, verify that your model is production-ready:

- **Safety gates pass** — the model meets all configured quality and safety thresholds
- **Benchmark scores meet minimums** — perplexity and accuracy are within acceptable ranges
- **Metadata is complete** — training configuration, dataset version, and evaluation results are recorded
- **Export succeeds** — the model loads correctly in the target format and produces expected outputs

Forge bundles these checks into the \`forge verify\` command so you can run them as a single pre-deployment step.
`,
};
