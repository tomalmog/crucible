import type { ReactNode } from "react";
import type { InterpTab } from "./interpTabs";
import { ModelAnatomyView } from "./ModelAnatomyView";
import type { ModelAnatomyData } from "./modelAnatomyTypes";

interface InterpHeroProps {
  anatomy: ModelAnatomyData | null;
  onSelect: (tab: InterpTab) => void;
}

export function InterpHero({ anatomy, onSelect }: InterpHeroProps): ReactNode {
  return (
    <section className="interp-hero">
      <div className="interp-hero-copy">
        <span className="interp-kicker">Mechanistic interpretability</span>
        <h1>Trace model behavior through the residual stream.</h1>
        <p>
          Crucible turns completed interp jobs into a compact evidence board:
          token predictions, layer geometry, causal recovery, sparse features,
          and steering effects stay tied to the model that produced them.
        </p>
        <div className="interp-hero-actions">
          <button type="button" className="btn btn-primary" onClick={() => onSelect("sae")}>
            Inspect sparse features
          </button>
          <button type="button" className="btn" onClick={() => onSelect("activation-patching")}>
            Open patching map
          </button>
        </div>
        <div className="interp-agent-brief" aria-label="Investigation runbook">
          <span>Investigation runbook</span>
          <ol>
            <li><strong>Localize</strong><small>logit lens and PCA</small></li>
            <li><strong>Test causality</strong><small>activation patching</small></li>
            <li><strong>Extract features</strong><small>SAE and steering</small></li>
          </ol>
        </div>
      </div>
      <ModelAnatomyView data={anatomy} onSelect={onSelect} />
    </section>
  );
}
