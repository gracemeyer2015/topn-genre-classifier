# Experiment notes

## Hypothesis
Split the difference between dropout 0 and dropout 0.5. Try 0.25 with the
same GAP architecture and see if it lands in a better spot than either
extreme.

## Result summary
By epoch 15 train accuracy is 65.0% and val accuracy is 64.0%. Val loss ends
at 1.15. For almost the entire run val accuracy stayed equal to or ahead of
train accuracy. Only at the very last epoch did train edge slightly ahead,
by about 1 point. Val loss kept trending down the whole run with some small
bumps, no real plateau yet and no upward turn either.

## Interpretation
This looks like the best balance of the three dropout settings so far. It
learns faster than dropout 0.5 and does not show the early overfitting
crossover that dropout 0 started to show. Train and val stayed close
together almost the whole time instead of one pulling away from the other.
It also had not finished improving by epoch 15, both losses were still
moving, so it may do even better with more epochs.

## Next experiment
Train this same config, dropout 0.25 with GAP, for more epochs, maybe 30.
Since it has not leveled off yet and is not showing overfitting, more
training should help without hurting it. If it stays healthy and levels off
around a good val accuracy, this becomes the config to lock in.
