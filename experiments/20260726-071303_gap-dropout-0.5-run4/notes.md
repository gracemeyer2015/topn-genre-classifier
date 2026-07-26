# Experiment notes

## Hypothesis
Repeat run of plain dropout 0.5 (no batch norm), same settings as the other runs in this group,
new random start. Part of bringing this setting up to 10 total runs so we
can compare group averages instead of trusting a single noisy run.

## Result summary
Final val accuracy 68.9%, final train accuracy 66.0%,
gap -2.9 points. Final val loss 1.024.

## Interpretation
One data point in the repeated-trial comparison for plain dropout 0.5 (no batch norm). See the
group average across all 10 runs of this setting for the real conclusion,
a single run's number isn't trustworthy on its own.

## Next experiment
Once all 4 settings (dropout 0.25/0.5, with/without batch norm) reach 10
runs each, compare the group averages to pick the setting to lock in.
