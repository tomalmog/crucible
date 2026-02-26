export function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function repoAuthor(repoId: string, author: string): string {
  if (author) return author;
  const slash = repoId.indexOf("/");
  return slash > 0 ? repoId.slice(0, slash) : "";
}

export function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export const MODEL_DISPLAY_TAGS = [
  "transformers", "safetensors", "pytorch", "gguf", "onnx",
  "text-generation", "text-to-image", "conversational",
  "fill-mask", "question-answering", "summarization", "translation",
];

export const DATASET_DISPLAY_TAGS = [
  "format:parquet", "modality:text", "modality:tabular", "modality:image",
  "library:datasets", "license:mit", "license:apache-2.0", "license:cc-by-4.0",
];

export function visibleTags(tags: string[], allowList: string[], max: number = 5): string[] {
  return tags.filter((t) => allowList.includes(t)).slice(0, max);
}

export function sizeTag(tags: string[]): string | null {
  const match = tags.find((t) => t.startsWith("size_categories:"));
  return match ? match.replace("size_categories:", "") : null;
}
