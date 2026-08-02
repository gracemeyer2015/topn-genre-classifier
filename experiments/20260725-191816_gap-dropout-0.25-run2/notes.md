# Experiment notes

## Hypothesis
Repeat the dropout 0.25 run again with a new random start. We never fixed a
random seed, so each run starts from different initial weights. Running the
same settings again checks whether dropout 0.25 beating dropout 0.5 last
time was a real pattern or just luck from one run.

## Result summary
Final val accuracy 71.8%, best val accuracy also 71.8% at the last epoch.
Train and val stay close together, final gap only 1.0 point, the smallest
gap seen in any dropout 0.25 run so far.

## Interpretation
This run did even better than the first dropout 0.25 run and kept a small
gap. Good sign that dropout 0.25 is a solid, repeatable choice and not a
lucky one-off.

## Next experiment
Run it one more time (run3) and also repeat dropout 0.5 twice, so we have 3
runs of each to compare fairly.
