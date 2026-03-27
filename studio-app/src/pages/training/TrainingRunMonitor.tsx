/**
 * Training progress parsing utilities.
 *
 * Used by UnifiedJobRow to extract epoch/batch/loss info from
 * training stdout for inline display on the Jobs page.
 */

export interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  batch?: number;
  totalBatches?: number;
  loss?: number;
  validationLoss?: number;
  etaSeconds?: number;
  meanReward?: number;
}

export function parseTrainingProgress(stdout: string): TrainingProgress | null {
  const lines = stdout.split("\n");
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line.startsWith("{")) continue;
    try {
      const parsed = JSON.parse(line);
      if (
        parsed.event === "training_epoch_completed" ||
        parsed.event === "training_batch_progress"
      ) {
        return {
          epoch: parsed.epoch,
          totalEpochs: parsed.total_epochs,
          batch: parsed.batch,
          totalBatches: parsed.total_batches,
          loss: parsed.train_loss ?? parsed.loss,
          validationLoss: parsed.validation_loss,
          etaSeconds: parsed.eta_seconds,
          meanReward: parsed.mean_reward,
        };
      }
    } catch {
      continue;
    }
  }
  return null;
}
