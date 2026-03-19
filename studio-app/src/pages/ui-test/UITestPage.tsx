import { useState } from "react";
import { PageHeader } from "../../components/shared/PageHeader";
import { MetricCard } from "../../components/shared/MetricCard";
import { BarChart } from "../../components/shared/BarChart";
import { FormField } from "../../components/shared/FormField";
import { FormSection } from "../../components/shared/FormSection";
import { TabBar } from "../../components/shared/TabBar";
import { CommandProgress } from "../../components/shared/CommandProgress";
import { ListRow } from "../../components/shared/ListRow";
import { ConfirmDeleteModal } from "../../components/shared/ConfirmDeleteModal";
import { StatusConsole } from "../../components/shared/StatusConsole";
import { DetailPage } from "../../components/shared/DetailPage";
import {
  Trash2, Plus, Download, Loader2, Inbox, AlertTriangle,
  Star, Settings, FileText, Heart, ArrowDown, BookOpen,
  FolderOpen, ChevronDown, X, Check,
} from "lucide-react";

const DEMO_TABS = ["overview", "details", "settings"] as const;
type DemoTab = (typeof DEMO_TABS)[number];

export function UITestPage() {
  const [inputValue, setInputValue] = useState("");
  const [inputError, setInputError] = useState("");
  const [selectValue, setSelectValue] = useState("option1");
  const [textareaValue, setTextareaValue] = useState("");
  const [checkA, setCheckA] = useState(false);
  const [checkB, setCheckB] = useState(true);
  const [checkDisabled] = useState(false);
  const [activeTab, setActiveTab] = useState<DemoTab>("overview");
  const [showModal, setShowModal] = useState(false);
  const [showDetail, setShowDetail] = useState(false);
  const [pathValue, setPathValue] = useState("");
  const [activePalette, setActivePalette] = useState("copper");

  if (showDetail) {
    return (
      <DetailPage title="Model Details" onBack={() => setShowDetail(false)} actions={<button className="btn btn-sm">Export</button>}>
        <div className="panel">
          <p className="text-sm" style={{ padding: 16 }}>Detail page content goes here. Click "Back" to return.</p>
        </div>
      </DetailPage>
    );
  }

  return (
    <>
      <PageHeader title="UI Library" />

      {/* ══════════════════════════════════════════
          PRIMITIVES
          ══════════════════════════════════════════ */}

      {/* ─── Buttons ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Buttons</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Variants</p>
          <div className="flex-row" style={{ gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            <button className="btn">Default</button>
            <button className="btn btn-primary">Primary</button>
            <button className="btn btn-ghost">Ghost</button>
            <button className="btn btn-success">Success</button>
            <button className="btn btn-error">Error</button>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Sizes</p>
          <div className="flex-row" style={{ gap: 8, alignItems: "center", marginBottom: 16 }}>
            <button className="btn btn-sm">Small</button>
            <button className="btn">Medium</button>
            <button className="btn btn-lg">Large</button>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>With Icons</p>
          <div className="flex-row" style={{ gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            <button className="btn btn-primary"><Plus size={14} /> Create</button>
            <button className="btn btn-error"><Trash2 size={14} /> Delete</button>
            <button className="btn"><Download size={14} /> Download</button>
            <button className="btn btn-sm"><Plus size={12} /> Small with Icon</button>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Icon Only</p>
          <div className="flex-row" style={{ gap: 8, marginBottom: 16 }}>
            <button className="btn btn-icon"><Plus size={16} /></button>
            <button className="btn btn-ghost btn-icon"><Trash2 size={16} /></button>
            <button className="btn btn-sm btn-icon"><Plus size={14} /></button>
            <button className="btn btn-ghost btn-sm btn-icon"><Trash2 size={12} /></button>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>States</p>
          <div className="flex-row" style={{ gap: 8, flexWrap: "wrap" }}>
            <button className="btn btn-primary" disabled>Disabled</button>
            <button className="btn btn-primary"><Loader2 size={14} className="spin" /> Loading</button>
            <button className="btn" disabled>Disabled Default</button>
            <button className="btn btn-ghost" disabled>Disabled Ghost</button>
          </div>
        </div>
      </div>

      {/* ─── Badges ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Badges</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="flex-row" style={{ gap: 8, flexWrap: "wrap" }}>
            <span className="badge">Default</span>
            <span className="badge badge-accent">Accent</span>
            <span className="badge badge-success">Success</span>
            <span className="badge badge-warning">Warning</span>
            <span className="badge badge-error">Error</span>
          </div>
        </div>
      </div>

      {/* ─── Input ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Input</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <div className="ff">
              <label className="ff-label" htmlFor="ui-input-default">Default Input</label>
              <input id="ui-input-default" type="text" placeholder="Type something..." value={inputValue} onChange={(e) => setInputValue(e.target.value)} />
            </div>
            <div className="ff">
              <label className="ff-label" htmlFor="ui-input-required">Required Input <span className="ff-required">*</span></label>
              <input id="ui-input-required" type="text" placeholder="This field is required" />
              <span className="ff-hint">Helper text goes here.</span>
            </div>
            <div className="ff">
              <label className="ff-label" htmlFor="ui-input-error">Input with Error</label>
              <input id="ui-input-error" type="text" placeholder="Invalid value" value={inputError} onChange={(e) => setInputError(e.target.value)} style={{ borderColor: "var(--error)" }} />
              <span className="error-text">This field has an error.</span>
            </div>
            <div className="ff">
              <label className="ff-label" htmlFor="ui-input-disabled">Disabled Input</label>
              <input id="ui-input-disabled" type="text" placeholder="Cannot edit" disabled />
            </div>
          </div>
        </div>
      </div>

      {/* ─── Select ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Select</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <div className="ff">
              <label className="ff-label" htmlFor="ui-select-default">Default Select</label>
              <select id="ui-select-default" value={selectValue} onChange={(e) => setSelectValue(e.target.value)}>
                <option value="option1">Option 1</option>
                <option value="option2">Option 2</option>
                <option value="option3">Option 3</option>
              </select>
            </div>
            <div className="ff">
              <label className="ff-label" htmlFor="ui-select-disabled">Disabled Select</label>
              <select id="ui-select-disabled" disabled><option>Cannot change</option></select>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Textarea ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Textarea</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <div className="ff">
              <label className="ff-label" htmlFor="ui-textarea-default">Default Textarea</label>
              <textarea id="ui-textarea-default" rows={3} placeholder="Enter multiple lines..." value={textareaValue} onChange={(e) => setTextareaValue(e.target.value)} />
            </div>
            <div className="ff">
              <label className="ff-label" htmlFor="ui-textarea-disabled">Disabled Textarea</label>
              <textarea id="ui-textarea-disabled" rows={3} placeholder="Cannot edit" disabled />
            </div>
          </div>
        </div>
      </div>

      {/* ─── Checkbox ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Checkbox</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
              <input type="checkbox" checked={checkA} onChange={(e) => setCheckA(e.target.checked)} style={{ width: 16, height: 16, margin: 0 }} />
              <span className="text-sm">Unchecked by default</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
              <input type="checkbox" checked={checkB} onChange={(e) => setCheckB(e.target.checked)} style={{ width: 16, height: 16, margin: 0 }} />
              <span className="text-sm">Checked by default</span>
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "not-allowed", opacity: 0.4 }}>
              <input type="checkbox" checked={checkDisabled} disabled style={{ width: 16, height: 16, margin: 0 }} />
              <span className="text-sm">Disabled</span>
            </label>
          </div>
        </div>
      </div>

      {/* ─── PathInput ─── */}
      <div className="panel">
        <div className="panel-header"><h3>PathInput</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <div className="ff">
              <label className="ff-label">File Path</label>
              <div className="path-input">
                <input value={pathValue} onChange={(e) => setPathValue(e.target.value)} placeholder="/path/to/model" />
                <button type="button" className="btn btn-ghost btn-sm path-input-browse" title="Browse...">
                  <FolderOpen size={14} />
                </button>
              </div>
            </div>
            <div className="ff">
              <label className="ff-label">Disabled Path</label>
              <div className="path-input">
                <input value="" placeholder="/locked/path" disabled />
                <button type="button" className="btn btn-ghost btn-sm path-input-browse" disabled title="Browse...">
                  <FolderOpen size={14} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── SearchableDropdown ─── */}
      <div className="panel">
        <div className="panel-header"><h3>SearchableDropdown</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Static demo (DatasetSelect / ModelSelect pattern)</p>
          <div className="dataset-select" style={{ maxWidth: 360 }}>
            <div className="dataset-select-input-wrap">
              <input value="my-training-data" readOnly placeholder="Select a dataset" />
              <button type="button" className="dataset-select-clear"><X size={14} /></button>
            </div>
          </div>
          <div style={{ marginTop: 12 }}>
            <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Dropdown open (mock)</p>
            <div style={{ maxWidth: 360, position: "relative" }}>
              <div className="dataset-select-input-wrap" style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-md)", background: "var(--bg-input)" }}>
                <input value="" placeholder="Search..." readOnly style={{ border: "none", boxShadow: "none" }} />
                <ChevronDown size={14} className="dataset-select-chevron" />
              </div>
              <ul className="dataset-select-dropdown" style={{ position: "relative", marginTop: 4 }}>
                <li className="dataset-select-header">Local</li>
                <li><button type="button" className="dataset-select-option">my-training-data</button></li>
                <li><button type="button" className="dataset-select-option">eval-holdout</button></li>
                <li className="dataset-select-header">Remote</li>
                <li><button type="button" className="dataset-select-option">cluster-dataset-v2</button></li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* ─── FilterBar ─── */}
      <div className="panel">
        <div className="panel-header"><h3>FilterBar</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="filter-bar">
            <select defaultValue="all"><option value="all">All Status</option><option value="running">Running</option><option value="completed">Completed</option><option value="failed">Failed</option></select>
            <select defaultValue="all"><option value="all">All Locations</option><option value="local">Local</option><option value="remote">Remote</option></select>
            <select defaultValue="all"><option value="all">All Types</option><option value="training">Training</option><option value="eval">Eval</option><option value="sweep">Sweep</option></select>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          DATA DISPLAY
          ══════════════════════════════════════════ */}

      {/* ─── MetricCard ─── */}
      <div className="panel">
        <div className="panel-header"><h3>MetricCard</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>stats-grid (auto-fit)</p>
          <div className="stats-grid">
            <MetricCard label="Accuracy" value="94.2%" />
            <MetricCard label="Loss" value="0.0312" />
            <MetricCard label="Epochs" value="5" />
            <MetricCard label="GPU Memory" value="24 GB" />
            <MetricCard label="Accelerator" value="cuda" />
          </div>
        </div>
      </div>

      {/* ─── OverviewTable ─── */}
      <div className="panel">
        <div className="panel-header"><h3>OverviewTable</h3></div>
        <table className="overview-table">
          <tbody>
            <tr><td className="overview-label">Model Name</td><td className="overview-value">llama-3-8b-instruct</td></tr>
            <tr><td className="overview-label">Architecture</td><td className="overview-value">LlamaForCausalLM</td></tr>
            <tr><td className="overview-label">Parameters</td><td className="overview-value">8,030,261,248</td></tr>
            <tr><td className="overview-label">Precision</td><td className="overview-value">bf16</td></tr>
            <tr><td className="overview-label">Status</td><td className="overview-value"><span className="badge badge-success">Ready</span></td></tr>
          </tbody>
        </table>
      </div>

      {/* ─── RuntimeKeyValue ─── */}
      <div className="panel">
        <div className="panel-header"><h3>RuntimeKeyValue</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <dl className="runtime-key-value-list">
            <div className="runtime-key-value-row"><dt>accelerator</dt><dd>cuda</dd></div>
            <div className="runtime-key-value-row"><dt>gpu_count</dt><dd>2</dd></div>
            <div className="runtime-key-value-row"><dt>recommended_precision_mode</dt><dd>bf16</dd></div>
            <div className="runtime-key-value-row"><dt>gpu_name</dt><dd>NVIDIA A100-SXM4-80GB</dd></div>
            <div className="runtime-key-value-row"><dt>cpu_count</dt><dd>64</dd></div>
          </dl>
        </div>
      </div>

      {/* ─── BarChart ─── */}
      <div className="panel">
        <div className="panel-header"><h3>BarChart</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <BarChart rows={[
            { label: "English", value: 4500 },
            { label: "Python", value: 2100 },
            { label: "JavaScript", value: 1800 },
            { label: "Markdown", value: 900 },
            { label: "Other", value: 350 },
          ]} />
        </div>
      </div>

      {/* ─── EmptyState ─── */}
      <div className="panel">
        <div className="panel-header"><h3>EmptyState</h3></div>
        <div className="empty-state">
          <div className="empty-state-icon"><Inbox /></div>
          <h3>No datasets found</h3>
          <p>Ingest your first dataset to get started with training.</p>
        </div>
      </div>

      {/* ─── SampleCard ─── */}
      <div className="panel">
        <div className="panel-header"><h3>SampleCard</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="grid-2">
            <div className="sample-card">
              <header>
                <span>rec-001</span>
                <span className="badge badge-accent">en</span>
                <span>0.92</span>
              </header>
              <p>The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow models to weigh the importance of different parts...</p>
              <small>source: arxiv/2017.01234</small>
            </div>
            <div className="sample-card">
              <header>
                <span>rec-002</span>
                <span className="badge badge-accent">py</span>
                <span>0.87</span>
              </header>
              <p>def train_model(config: TrainingConfig) -&gt; Model: optimizer = AdamW(model.parameters(), lr=config.learning_rate) for epoch in range(config.num_epochs):</p>
              <small>source: github/example-repo</small>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          CARDS
          ══════════════════════════════════════════ */}

      {/* ─── MethodCard ─── */}
      <div className="panel">
        <div className="panel-header"><h3>MethodCard</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="method-grid">
            <div className="method-card">
              <div className="method-card-header">
                <span className="method-card-name">SFT</span>
                <span className="method-card-tag">Fine-Tuning</span>
              </div>
              <span className="method-card-description">Standard supervised fine-tuning on instruction-response pairs.</span>
              <div className="method-card-footer">
                <button className="method-card-docs"><BookOpen size={12} /> Docs</button>
              </div>
            </div>
            <div className="method-card">
              <div className="method-card-header">
                <span className="method-card-name">DPO</span>
                <span className="method-card-tag">Alignment</span>
              </div>
              <span className="method-card-description">Direct Preference Optimization for aligning models with human preferences.</span>
              <div className="method-card-footer">
                <button className="method-card-docs"><BookOpen size={12} /> Docs</button>
              </div>
            </div>
            <div className="method-card">
              <div className="method-card-header">
                <span className="method-card-name">LoRA</span>
                <span className="method-card-tag">Efficient</span>
              </div>
              <span className="method-card-description">Low-Rank Adaptation for parameter-efficient fine-tuning with minimal memory.</span>
              <div className="method-card-footer">
                <button className="method-card-docs"><BookOpen size={12} /> Docs</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── HubCard ─── */}
      <div className="panel">
        <div className="panel-header"><h3>HubCard</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="hub-grid">
            <div className="hub-card">
              <div className="hub-card-header">
                <div>
                  <div className="hub-card-repo">meta-llama/Llama-3-8B</div>
                  <div className="hub-card-author">meta-llama</div>
                </div>
                <span className="hub-card-task">text-generation</span>
              </div>
              <div className="hub-card-stats">
                <span className="hub-card-stat"><Heart size={13} /> 12.4k</span>
                <span className="hub-card-stat"><ArrowDown size={13} /> 890k</span>
              </div>
              <div className="hub-card-bottom">
                <span className="hub-card-date">Updated 3 days ago</span>
              </div>
            </div>
            <div className="hub-card">
              <div className="hub-card-header">
                <div>
                  <div className="hub-card-repo">microsoft/phi-3-mini</div>
                  <div className="hub-card-author">microsoft</div>
                </div>
                <span className="hub-card-task">text-generation</span>
              </div>
              <div className="hub-card-stats">
                <span className="hub-card-stat"><Heart size={13} /> 3.2k</span>
                <span className="hub-card-stat"><ArrowDown size={13} /> 420k</span>
              </div>
              <div className="hub-card-bottom">
                <span className="hub-card-date">Updated 1 week ago</span>
              </div>
            </div>
            <div className="hub-card">
              <div className="hub-card-header">
                <div>
                  <div className="hub-card-repo">google/gemma-2-9b</div>
                  <div className="hub-card-author">google</div>
                </div>
                <span className="hub-card-task">text-generation</span>
              </div>
              <div className="hub-card-stats">
                <span className="hub-card-stat"><Heart size={13} /> 5.1k</span>
                <span className="hub-card-stat"><ArrowDown size={13} /> 650k</span>
              </div>
              <div className="hub-card-bottom">
                <span className="hub-card-date">Updated 2 weeks ago</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── RunRow ─── */}
      <div className="panel">
        <div className="panel-header"><h3>RunRow</h3></div>
        <div className="panel panel-flush">
          <div className="run-row" style={{ borderBottom: "1px solid var(--border-light)" }}>
            <div className="run-row-header">
              <div className="flex-row">
                <span className="run-row-id">run-20260318T102300Z-a1b2c3</span>
                <span className="badge badge-success">completed</span>
              </div>
              <div className="flex-row">
                <button className="btn btn-ghost btn-sm btn-icon"><Trash2 size={12} /></button>
              </div>
            </div>
            <div className="run-row-meta">
              <span>my-dataset</span>
              <span>2 min ago</span>
            </div>
            <span className="run-row-path">/output/checkpoint-final</span>
          </div>
          <div className="run-row" style={{ borderBottom: "1px solid var(--border-light)" }}>
            <div className="run-row-header">
              <div className="flex-row">
                <span className="run-row-id">run-20260317T154500Z-d4e5f6</span>
                <span className="badge badge-accent">running</span>
              </div>
            </div>
            <div className="run-row-meta">
              <span>eval-holdout</span>
              <span>5 min ago</span>
            </div>
          </div>
          <div className="run-row">
            <div className="run-row-header">
              <div className="flex-row">
                <span className="run-row-id">run-20260316T091200Z-g7h8i9</span>
                <span className="badge badge-error">failed</span>
              </div>
            </div>
            <div className="run-row-meta">
              <span>code-instruct</span>
              <span>yesterday</span>
            </div>
          </div>
        </div>
      </div>

      {/* ─── ModelVersionRow ─── */}
      <div className="panel">
        <div className="panel-header"><h3>ModelVersionRow</h3></div>
        <div className="panel panel-flush">
          <div style={{ borderBottom: "1px solid var(--border-light)", padding: "12px 18px" }}>
            <div className="model-version-row">
              <div className="model-version-meta">
                <span className="text-sm" style={{ fontWeight: 500 }}>v3 (latest)</span>
                <span className="text-xs text-tertiary">8.2 GB</span>
              </div>
              <span className="text-xs text-tertiary">Created 2 hours ago</span>
            </div>
          </div>
          <div style={{ borderBottom: "1px solid var(--border-light)", padding: "12px 18px" }}>
            <div className="model-version-row">
              <div className="model-version-meta">
                <span className="text-sm" style={{ fontWeight: 500 }}>v2</span>
                <span className="text-xs text-tertiary">8.1 GB</span>
              </div>
              <span className="text-xs text-tertiary">Created 3 days ago</span>
            </div>
          </div>
          <div style={{ padding: "12px 18px" }}>
            <div className="model-version-row">
              <div className="model-version-meta">
                <span className="text-sm" style={{ fontWeight: 500 }}>v1</span>
                <span className="text-xs text-tertiary">7.9 GB</span>
              </div>
              <span className="text-xs text-tertiary">Created 1 week ago</span>
            </div>
          </div>
        </div>
      </div>

      {/* ─── PaletteSwatch ─── */}
      <div className="panel">
        <div className="panel-header"><h3>PaletteSwatch</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="palette-grid">
            {[
              { name: "copper", colors: ["#c77d4a", "#f5e6d3"] },
              { name: "slate", colors: ["#64748b", "#f1f5f9"] },
              { name: "emerald", colors: ["#059669", "#ecfdf5"] },
              { name: "violet", colors: ["#7c3aed", "#f5f3ff"] },
            ].map((p) => (
              <div
                key={p.name}
                className={`palette-swatch${activePalette === p.name ? " palette-swatch--active" : ""}`}
                onClick={() => setActivePalette(p.name)}
              >
                {activePalette === p.name && (
                  <div className="palette-swatch-check"><Check size={12} /></div>
                )}
                <div className="palette-swatch-colors">
                  <div style={{ background: p.colors[0] }} />
                  <div style={{ background: p.colors[1] }} />
                </div>
                <span className="palette-swatch-label">{p.name}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          LAYOUT
          ══════════════════════════════════════════ */}

      {/* ─── Panel ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Panel</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Default panel with header + action</p>
          <div className="panel" style={{ marginBottom: 12 }}>
            <div className="panel-header">
              <h3>Panel Title</h3>
              <button className="btn btn-sm">Action</button>
            </div>
            <p className="text-sm" style={{ padding: "0 16px 16px" }}>Panel content goes here.</p>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Flush panel</p>
          <div className="panel panel-flush">
            <div className="panel-header"><h3>Flush Panel</h3></div>
            <p className="text-sm" style={{ padding: "0 16px 16px" }}>Used for lists that go edge-to-edge.</p>
          </div>
        </div>
      </div>

      {/* ─── PageHeader ─── */}
      <div className="panel">
        <div className="panel-header"><h3>PageHeader</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <PageHeader title="Example Page">
            <button className="btn btn-sm"><Plus size={14} /> New Item</button>
            <button className="btn btn-sm btn-ghost"><Settings size={14} /></button>
          </PageHeader>
        </div>
      </div>

      {/* ─── DetailPage ─── */}
      <div className="panel">
        <div className="panel-header"><h3>DetailPage</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <button className="btn btn-sm" onClick={() => setShowDetail(true)}>Open Detail Page Demo</button>
        </div>
      </div>

      {/* ─── FormField ─── */}
      <div className="panel">
        <div className="panel-header"><h3>FormField (component)</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <FormField label="Model Name" required>
              <input type="text" placeholder="my-model" />
            </FormField>
            <FormField label="Learning Rate" hint="Recommended: 1e-5 to 5e-5">
              <input type="text" placeholder="0.00002" />
            </FormField>
            <FormField label="Output Format" required>
              <select><option>PyTorch (.pt)</option><option>SafeTensors (.safetensors)</option><option>ONNX (.onnx)</option></select>
            </FormField>
          </div>
        </div>
      </div>

      {/* ─── FormSection ─── */}
      <div className="panel">
        <div className="panel-header"><h3>FormSection</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <FormSection title="Advanced Options" defaultOpen>
            <div className="stack-sm">
              <FormField label="Gradient Accumulation Steps"><input type="number" placeholder="4" /></FormField>
              <FormField label="Warmup Steps"><input type="number" placeholder="100" /></FormField>
            </div>
          </FormSection>
          <FormSection title="Collapsed by Default">
            <p className="text-sm text-tertiary">This content is hidden until expanded.</p>
          </FormSection>
        </div>
      </div>

      {/* ─── TabBar ─── */}
      <div className="panel">
        <div className="panel-header"><h3>TabBar</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <TabBar tabs={DEMO_TABS} active={activeTab} onChange={setActiveTab} />
          <p className="text-sm">Active tab: <strong>{activeTab}</strong></p>
        </div>
      </div>

      {/* ─── WizardSteps ─── */}
      <div className="panel">
        <div className="panel-header"><h3>WizardSteps</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="wizard-steps">
            <span className="wizard-step">1. Select Method</span>
            <span className="wizard-step-separator">&gt;</span>
            <span className="wizard-step active">2. Configure</span>
            <span className="wizard-step-separator">&gt;</span>
            <span className="wizard-step">3. Review</span>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          FEEDBACK
          ══════════════════════════════════════════ */}

      {/* ─── ProgressBar ─── */}
      <div className="panel">
        <div className="panel-header"><h3>ProgressBar</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack">
            <CommandProgress label="Training epoch 3/5" percent={60} elapsed={120} remaining={80} />
            <CommandProgress label="Downloading model" percent={25} />
            <CommandProgress label="Complete" percent={100} elapsed={300} remaining={0} />
          </div>
        </div>
      </div>

      {/* ─── Console ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Console</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Default</p>
          <pre className="console" style={{ marginBottom: 16 }}>
{`[2026-03-18 10:23:01] Starting training run...
[2026-03-18 10:23:01] Loading dataset: my-dataset (4,500 records)
[2026-03-18 10:23:02] Model: llama-3-8b-instruct
[2026-03-18 10:23:02] Method: sft | Epochs: 3 | LR: 2e-5
[2026-03-18 10:23:03] Epoch 1/3 - Batch 1/120 - Loss: 2.4531`}
          </pre>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>Short variant</p>
          <pre className="console console-short">
{`$ crucible train --model ./my-model --dataset my-data
Training complete. Model saved to ./output/checkpoint-final`}
          </pre>
        </div>
      </div>

      {/* ─── StatusConsole (component) ─── */}
      <div className="panel">
        <div className="panel-header"><h3>StatusConsole (component)</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <StatusConsole title="Command Output" output={"Ingesting 1,200 records...\nValidating schema... OK\nWriting to lance format... Done.\nDataset 'my-data' created successfully."} />
        </div>
      </div>

      {/* ─── ErrorAlert ─── */}
      <div className="panel">
        <div className="panel-header"><h3>ErrorAlert</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <p className="text-xs text-tertiary">Inline error text</p>
            <span className="error-text">Something went wrong.</span>
            <p className="text-xs text-tertiary" style={{ marginTop: 8 }}>Error alert block</p>
            <div className="error-alert">CrucibleTrainingError: CUDA out of memory. Tried to allocate 2.00 GiB.</div>
            <p className="text-xs text-tertiary" style={{ marginTop: 8 }}>Prominent error alert</p>
            <div className="error-alert-prominent">
              <div className="flex-row" style={{ gap: 8, marginBottom: 4 }}>
                <AlertTriangle size={14} />
                <strong style={{ fontSize: "0.8125rem" }}>Training Failed</strong>
              </div>
              <p style={{ margin: 0, fontSize: "0.8125rem", color: "var(--error)" }}>Model checkpoint is corrupted. Please restart training from scratch.</p>
            </div>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════
          OVERLAYS
          ══════════════════════════════════════════ */}

      {/* ─── Modal ─── */}
      <div className="panel">
        <div className="panel-header"><h3>ConfirmDeleteModal</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <button className="btn btn-error btn-sm" onClick={() => setShowModal(true)}>
            <Trash2 size={12} /> Open Delete Modal
          </button>
        </div>
      </div>

      {showModal && (
        <ConfirmDeleteModal
          title="Delete Model"
          itemName="llama-3-8b-instruct"
          description="This action cannot be undone."
          isDeleting={false}
          onConfirm={() => setShowModal(false)}
          onCancel={() => setShowModal(false)}
        />
      )}

      {/* ══════════════════════════════════════════
          COMPOSITE
          ══════════════════════════════════════════ */}

      {/* ─── ListRow ─── */}
      <div className="panel">
        <div className="panel-header"><h3>ListRow</h3></div>
        <div className="panel panel-flush">
          <ListRow
            name="my-fine-tuned-model"
            meta={<><span>8.2 GB</span><span className="badge badge-success">Ready</span></>}
          />
          <ListRow
            name="distilled-small-v2"
            meta={<><span>1.4 GB</span><span className="badge badge-accent">Training</span></>}
            actions={<button className="btn btn-ghost btn-sm btn-icon"><Star size={12} /></button>}
          />
          <ListRow
            name="base-model-checkpoint"
            meta={<span>24.1 GB</span>}
            showChevron={false}
            actions={<>
              <button className="btn btn-ghost btn-sm btn-icon"><FileText size={12} /></button>
              <button className="btn btn-ghost btn-sm btn-icon"><Trash2 size={12} /></button>
            </>}
          />
        </div>
      </div>

      {/* ─── RegistryRow ─── */}
      <div className="panel">
        <div className="panel-header"><h3>RegistryRow</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="flex-row" style={{ alignItems: "center", padding: "4px 8px", gap: 8 }}>
            <span className="text-sm" style={{ flex: 1 }}>my-training-data</span>
            <span className="text-xs text-tertiary">35.0 KB</span>
            <div style={{ display: "flex", gap: 2 }}>
              <button className="btn btn-ghost btn-sm btn-icon"><Download size={12} /></button>
              <button className="btn btn-ghost btn-sm btn-icon"><Trash2 size={12} /></button>
            </div>
          </div>
          <div className="flex-row active" style={{ alignItems: "center", padding: "4px 8px", gap: 8 }}>
            <span className="text-sm" style={{ flex: 1 }}>eval-holdout</span>
            <span className="text-xs text-tertiary">12.8 KB</span>
            <div style={{ display: "flex", gap: 2 }}>
              <button className="btn btn-ghost btn-sm btn-icon"><Download size={12} /></button>
              <button className="btn btn-ghost btn-sm btn-icon"><Trash2 size={12} /></button>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Text Utilities ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Text Utilities</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <div className="stack-sm">
            <p className="text-xs">text-xs (0.6875rem)</p>
            <p className="text-sm">text-sm (0.75rem)</p>
            <p>Default body text</p>
            <p className="text-secondary">text-secondary</p>
            <p className="text-tertiary">text-tertiary</p>
            <p className="text-mono">text-mono (monospace)</p>
          </div>
        </div>
      </div>

      {/* ─── Layout Utilities ─── */}
      <div className="panel">
        <div className="panel-header"><h3>Layout Utilities</h3></div>
        <div style={{ padding: "0 16px 16px" }}>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>flex-row</p>
          <div className="flex-row" style={{ marginBottom: 16 }}>
            <span className="badge">Item 1</span>
            <span className="badge badge-accent">Item 2</span>
            <span className="badge badge-success">Item 3</span>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>grid-2</p>
          <div className="grid-2" style={{ marginBottom: 16 }}>
            <div className="panel"><p className="text-sm" style={{ padding: 16 }}>Column 1</p></div>
            <div className="panel"><p className="text-sm" style={{ padding: 16 }}>Column 2</p></div>
          </div>
          <p className="text-xs text-tertiary" style={{ marginBottom: 8 }}>stack (vertical, 16px gap)</p>
          <div className="stack">
            <div className="panel"><p className="text-sm" style={{ padding: 16 }}>Stack item 1</p></div>
            <div className="panel"><p className="text-sm" style={{ padding: 16 }}>Stack item 2</p></div>
          </div>
        </div>
      </div>
    </>
  );
}
