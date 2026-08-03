# Experiment notes

## Hypothesis
Try weight decay (L2 regularization) as a different kind of regularizer than
dropout. Light amount, 0.0001, combined with dropout 0.25, to see if it adds
anything on top.

## Result summary
Final val accuracy 68.1%, best val accuracy 70.4% at epoch 29. Final train
accuracy 70.1%, gap about 2.0 points.

## Interpretation
Lands right in the same range as plain dropout 0.25's other runs, not a
clear improvement. Only one run so far though, so this could just be normal
noise, same caution as everything else tested once.

## Next experiment
Would need repeat runs to know if weight decay is really doing anything.
Not a priority right now since it doesn't look like a standout.
