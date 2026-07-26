# Experiment notes

## Hypothesis
Repeat dropout 0.5 with a new random start, same reason as the dropout 0.25
repeats. Check if its earlier result was typical or lucky.

## Result summary
Final val accuracy dropped to 63.6%, the weakest result of any run in this
whole sweep. Best val accuracy during the run was 67.6%, also the weakest
best-case seen. Final gap was small, only 0.9 points.

## Interpretation
This run shows dropout 0.5 is less consistent than it first looked. Its
train and val stay close together like before, but the actual accuracy it
reaches this time is clearly worse than any dropout 0.25 run so far. Small
gap is good, but a small gap around a weaker number is not actually better.

## Next experiment
Run dropout 0.5 one more time (run3) to get a third data point before
comparing averages across both settings.
