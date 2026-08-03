# Experiment notes

## Hypothesis
Run the current best-looking setting (dropout 0.25, no batch norm) for 50
epochs instead of 30, to see if it had more room to improve or had already
plateaued.

## Result summary
Final val accuracy 73.4% at epoch 50, still climbing at the very end
(72.5% at epoch 49 to 73.4% at epoch 50). Train accuracy 78.6%, gap 5.2
points. This is the highest val accuracy seen in the whole investigation
from a full, still-improving trajectory (not just a lucky single epoch).

## Interpretation
There was real room to keep improving past 30 epochs. The gap (5.2 points)
is a bit bigger than the 30 epoch runs but not alarming, still much smaller
than the original baseline's 30+ point gap. This is the most promising
single result so far, though it's only one run and should be repeated to
make sure it's not a fluke before trusting it fully.

## Next experiment
Repeat this same setting (dropout 0.25, 50 epochs, no batch norm) once or
twice more to confirm the higher accuracy is real and not a lucky run.
