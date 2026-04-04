export interface HubFileEntry {
  filename: string;
  size: number;
}

export interface HubModelDetail {
  repo_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  task: string;
  last_modified: string;
  created_at: string;
  license: string;
  base_model: string;
  library: string;
  gated: string | false;
  files: HubFileEntry[];
  total_size: number;
}

export interface HubDatasetDetail {
  repo_id: string;
  author: string;
  downloads: number;
  likes: number;
  tags: string[];
  task_categories: string[];
  last_modified: string;
  created_at: string;
  license: string;
  gated: string | false;
  files: HubFileEntry[];
  total_size: number;
  splits: string[];
}
