# Negative sampling audit and migration

Audited V3 main commit: `434701fd4851a0d30062025015e513f484975d95`.

## Data flow and cause

`utils.load_data` reads ratings and maps IDs. `SampleGenerator` binarizes positive ratings and performs leave-one-out splitting. Previously it built the negative pool from train-only interactions, so a held-out positive remained eligible. The pool feeds `store_all_train_data`, `store_pseudo_train_data`, and validation/test candidate construction. `trainer.run_experiment` directly calls `test_data` once and resamples training and pseudo negatives each round; it does not call `get_data`.

`utils.compute_metrics` concatenates negative candidates and the held-out positive, ranks rows, and counts matches by item ID. A duplicated held-out item can therefore be counted twice at sufficiently large K (Recall and NDCG can exceed 1), while at smaller K duplicated candidates and tie ordering can also alter ranks. Training assigns sampled negatives label zero, contradicting observed positive feedback. These mechanisms can bias scores in different directions; no quantitative correction or guaranteed improvement is justified without rerunning experiments.

V2 PR #3 was merged and explicitly switched to `preprocess_ratings`. Independently inspected V2's current default-branch `data.py` on 2026-10-08: it still uses `self.train_ratings`; the historical fix cannot be assumed present today.

## Minimal change

Construct the pool from all preprocessed observed interactions. Excluding all observed pairs is conservative, including any nonpositive input rows; positive labels for learning still come from the training split only. Held-out IDs are used solely to exclude known interactions, never as extra training positives. Use safe `.loc` binarization. Fail with user ID, available count, and requested counts when the pool cannot meet the configured train/evaluation counts without replacement. Do not truncate counts, sample known positives, or enable replacement. Model, distillation, metric implementation, and default hyperparameters are unchanged.

## CPU verification

Run `python -m unittest discover -s tests -v` (requires torch, pandas, matplotlib, numpy).
Five synthetic CPU tests passed: user-specific complete-interaction exclusion; ordinary and pseudo training negatives; unique test candidates and exact Recall/NDCG values; insufficient and exact-size pools; cache load path without overwrites. Temporarily routing the sampler back to train-only data causes regression failures, including held-out items in training and evaluation. No training or external dataset processing was performed.

## Existing files and regeneration

No dataset cache directory or `.npy` data files are tracked/present in this checkout; local historical caches cannot be certified from repository contents. `get_data` loads existing `train.npy`, `val.npy`, and `test.npy` whenever `train.npy` exists. Code changes do not invalidate those files. Its behavior remains unchanged and tests verify it does not overwrite existing caches.

Treat caches created by the old sampler as potentially contaminated. Before using them for new trustworthy results, preserve them and generate a separate versioned set from the same source ratings, ID mapping, split policy, seed and negative counts; audit zero-label user/item pairs and evaluation negatives against all observed interactions. If provenance and an audit prove a cache is clean, regeneration solely for this issue is unnecessary. Rebuild ordinary/pseudo sampled negatives and validation/test candidates that were produced by the old logic; raw ratings and semantic/pseudo embeddings need not be rebuilt solely for this sampling bug. Do not merely change the code and reuse old sampled-data files. No historical data was deleted or overwritten.

Separate pre-existing blocker: `validate_data` refers to undefined `self.val_ratings`; the douban/bookcrossing branch removes validation rows but still asserts train+test equals the full input. Thus uncached `get_data()`/these split paths already require a separate split-contract repair before a complete cache set can be regenerated. This change does not invent a validation protocol or silently change the experimental split. The regression cache test covers existing-cache loading, not uncached generation.

## Conclusions requiring renewed evidence

Re-evaluating old checkpoints on clean, shared candidates isolates evaluation contamination but cannot remove training contamination. Final model comparisons need retraining with corrected sampling and evaluating on identical clean candidate sets across methods and seeds. Recheck absolute Recall/NDCG, method ordering, distillation gains, large/small/proxy subgroup results, ablations, hyperparameter rankings, variance/significance and best-round selection (which currently depends on test Recall). Old experiment logs alone cannot establish whether each run actually sampled a held-out positive or used older clean caches. Existing results should remain preserved and marked with their sampling provenance until audited. This patch neither changes best-round selection nor claims to fix its separate use of the test set.
