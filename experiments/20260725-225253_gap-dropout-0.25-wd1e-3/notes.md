# Experiment notes

## Hypothesis
Try a stronger weight decay, 0.001, combined with dropout 0.25, to see if
more L2 regularization helps more than the lighter amount did.

## Result summary
Final val accuracy 67.6%, train accuracy 65.1%, gap actually negative
(-2.5 points, val ahead of train).

## Interpretation
Also lands in the normal range for dropout 0.25, not a standout either way.
Combined with the wd1e-4 result, weight decay doesn't look like it's adding
much on top of dropout here, matching what other groups (crlandsc) found
too, that L2 didn't help their CNN.

## Next experiment
Not a priority to pursue further given neither weight decay amount stood
out from plain dropout.
