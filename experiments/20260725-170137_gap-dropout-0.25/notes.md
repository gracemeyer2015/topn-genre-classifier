# Experiment notes

## Hypothesis
Run dropout 0.25 for 30 epochs instead of 15. At 15 epochs it had not
finished improving yet and showed no overfitting gap. Training longer
should show whether it keeps improving safely or eventually splits apart
like dropout 0 did.

## Result summary
Train accuracy ends at 72.2% and val accuracy ends at 69.6%, a gap of only
2.6 points. Val accuracy reaches as high as 70.0% more than once in the
back half of the run. Val loss keeps trending down overall, best point is
1.00 at epoch 29, ending at 1.03. The two lines stay close together the
entire run instead of splitting apart.

## Interpretation
This is almost the same final val accuracy as dropout 0 (69.6% vs 68.5%,
with dropout 0's best moment at 71.3%), but without the growing overfitting
gap. Dropout 0 needed to let train accuracy run up to 79.6% to get there,
which did not translate into better val performance, it just meant more
memorizing. Dropout 0.25 gets basically the same result more honestly, and
without the instability dropout 0 showed late in training. Per Andrew Ng's
Improving Deep Neural Networks course, this is the expected trade off: once
a model is not underfitting anymore, more capacity or more training epochs
mainly just risks overfitting rather than buying real accuracy.

## Next experiment
This looks like the config to lock in. Next step is comparing it against
dropout 0.5 at the same 30 epochs to confirm 0.25 is genuinely the better
middle ground, then saving a real model checkpoint from this config for
handoff.
