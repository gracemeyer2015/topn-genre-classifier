# Experiment notes

## Hypothesis
Add dropout (p=0.5) right before the final Linear layer, with everything
else identical to baseline, to isolate how much dropout alone helps.

## Result summary
Training accuracy still climbs high (97%) but noticeably slower than
baseline's ~99%+. Validation loss's best point is epoch 3 (0.99), which is
better than baseline's best point (1.05 at epoch 3). Validation accuracy
ends the run at 72.5%, higher than baseline's final 68.7%, and generally
trends upward more over the back half of training instead of just bouncing
around. Validation loss still rises after epoch 3, but much more gently than
baseline did (ends around 1.6 instead of baseline's ~2.5) (see curves.png).

## Interpretation
Dropout does help. We see a better best-case val_loss, better final
val_acc, gentler overfitting curve. But it doesn't fully fix it: val_loss
still turns upward after epoch 3, and there's still a big gap between train
accuracy (97%) and val accuracy (72%). Note that dropout  does make it harder
for the model to rely on memorizing, but the final Linear layer still has
163,850 parameters and this is likely too many. Dropout won't shrink that, it just
randomly blocks some of it each pass.

## Next experiment
Global average pooling to shrink the flatten size (16,384 -> 64)
before the final Linear, combined with dropout. This attacks the actual
oversized layer instead of just making it harder to use.
