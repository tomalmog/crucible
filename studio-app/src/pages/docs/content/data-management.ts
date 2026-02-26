import type { DocEntry } from "../docsRegistry";

export const dataManagement: DocEntry = {
  slug: "data-management",
  title: "Data Management",
  category: "Data",
  content: `
## Data Management

Forge provides a four-step pipeline for managing training data: **ingest**, **version**, **filter**, and **export**.

### Ingest

Import data from local files into a Forge dataset.

\`\`\`bash
forge ingest --source <path> --name <dataset>
\`\`\`

Supported formats: **CSV**, **JSONL**, **Parquet**, and **plain text** files. Forge auto-detects the format from the file extension and validates each record on import. You can point at a single file or an entire directory — Forge will recursively discover and ingest all supported files.

### Versions

Every call to \`forge ingest\` creates a new immutable version of the dataset.

\`\`\`bash
forge versions --dataset <name>
\`\`\`

This shows the full version history: timestamps, record counts, and source paths. Versions are append-only — you never lose previous data. You can reference any version by its ID when filtering or exporting.

### Filter

Narrow a dataset down to the records you actually need for training.

\`\`\`bash
forge filter --dataset <name> --query <expression>
\`\`\`

Filter expressions support:

- **Field comparisons** — \`length > 512\`, \`label == "positive"\`
- **Text patterns** — \`text contains "machine learning"\`
- **Length constraints** — \`token_count between 100 and 2048\`

Filtering creates a new version so the original data is always preserved.

### Export

Export a dataset to a training-ready file format.

\`\`\`bash
forge export-training --dataset <name> --format jsonl --output <path>
\`\`\`

Supported export formats include **JSONL**, **CSV**, and **Parquet**. The exported file is ready to pass directly into \`forge train\` or any other training framework. You can also specify a version or apply filters inline during export.
`,
};
