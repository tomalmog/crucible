# Multi-Model Training: Location Mismatch

## Problem
RLHF and Distillation training methods accept multiple model inputs that could be in different locations (local vs remote):

- **RLHF**: policy model (required) + reward model (optional)
- **Distillation**: teacher model (required) + student model (optional)

If one model is local and the other is remote, training cannot proceed normally.

## Possible Solutions
1. **Cross-location communication**: Allow models on different hosts to talk to each other during training.
2. **Same-location requirement**: Require both models to be in the same location. Filter the secondary model dropdown to only show models matching the primary model's location.

## Status
Deferred. For now, RLHF and Distillation panels are excluded from the auto-location logic and keep their existing manual local/remote toggle.
