# Experiment notes

## Hypothesis
Test batch norm completely on its own, with dropout turned off entirely, to
see how much batch norm helps by itself without dropout in the picture.

## Result summary
Final val accuracy 72.3%, but train accuracy 85.9%, a gap of 13.6 points,
the largest gap seen among any GAP-based run, even worse than plain dropout
0 without batch norm (11.1 points).

## Interpretation
Batch norm without any dropout does not prevent overfitting here, it may
even make it slightly worse than no regularization at all. This matches
a known finding in machine learning research that dropout and batch norm
can interact poorly, and confirms dropout is still doing real, necessary
work that batch norm alone doesn't replace.

## Next experiment
Don't rely on batch norm alone as a regularizer. Keep dropout in the mix
for any batch norm experiments, which is what the rest of tonight's batch
already does.
