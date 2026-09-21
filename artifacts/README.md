# Artifacts

Training runs. Only the small JSON sidecars are tracked — weights (`*.keras`,
`*.h5`, `best.weights.h5`) and exported `.tflite` files are ignored, and the
`.tflite` should be regenerated with `make export RUN=<run>` from the checkpoint
rather than committed.

## Status: none of these runs is a valid result

Every run in this directory predates three data-pipeline fixes, so **do not quote
any metric here as keyword-detection performance**. They are kept because the
diagnostics in `history.json` are what found the bugs, and deleting the evidence
would make the fixes look arbitrary.

| Run | What it was | Why it is invalid |
| --- | --- | --- |
| `smoke2` | first smoke train | old keyword spans (positives placed on silence) |
| `smoke3` | smoke after the melspec fix | same |
| `smoke4` | smoke after deterministic eval | same |
| `abl-noaug` | augmentation ablation, `--augment none` | same + class-ordered batches |
| `abl-evalplace` | augmentation ablation, `--placement eval` | **trained on zero positives** (see below) |

### What went wrong

1. **Keyword spans were displaced.** `build_cache.speech_bounds` picked the
   *longest* loud run rather than the loudest moment, so on 1.7–6.4 s recordings
   it often selected background noise instead of the word. Positive-labelled
   windows then landed on silence. Fixed by anchoring the span on the peak frame
   and capping it at 1.1 s.

2. **Batches were sorted by class.** The TFRecord shards are written grouped by
   kind and the shuffle buffer was 1,000 elements, so the per-batch positive rate
   decayed from 158/256 to 26/256 across a pass. Batch normalisation therefore
   normalised per class instead of per batch. Fixed with a 40,000-element buffer.

3. **`--placement eval` trained on no positives at all.** The old
   `eval_aligned = slot < eval_slots // 2` rule is never true for
   `eval_slots == 1`, so that arm's training set contained only negatives.

### The one number worth remembering

`abl-evalplace` reached `val_auc` 0.8196 while being trained on **zero
positives**. A model that has never seen a keyword cannot detect one, so that
0.82 is a shortcut, not a skill: the validation set can be partly ranked by a
nuisance cue — loudness is the prime suspect, since the confusable-phrase
negatives measure roughly 5 dB louder than the keyword positives. Any future
evaluation has to be able to explain that number away, which is why
`ml/tools/evaluate.py` reports per-class rates rather than a single AUC.

## Regenerating

```bash
make data      # fetch corpora
make cache     # rebuild TFRecords with the corrected spans
make train     # writes a new artifacts/<run>/
```
