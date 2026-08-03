# Experiment notes

## Hypothesis
Third run of dropout 0.25, same settings, new random start. Third data point
to compare fairly against three runs of dropout 0.5.

## Result summary
Final val accuracy 67.9%, the lowest of the three dropout 0.25 runs, but
best val accuracy during the run was 71.5% at epoch 29. Final gap between
train and val was 6.1 points, the largest gap seen for dropout 0.25 so far.

## Interpretation
This run is noisier than the first two, showing dropout 0.25 is not
perfectly consistent every time. But even its weaker moments are still
close to what dropout 0.5 achieves at its best, so this does not undo the
overall pattern across all three runs.

## Next experiment
Compare all three dropout 0.25 runs against all three dropout 0.5 runs
together, using the average and spread, not just one run each, to decide
which setting to lock in.
