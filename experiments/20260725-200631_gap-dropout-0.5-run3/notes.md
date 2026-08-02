# Experiment notes

## Hypothesis
Third run of dropout 0.5, same settings, new random start. Third data point
to fairly compare against three runs of dropout 0.25.

## Result summary
Final val accuracy 70.2%, best val accuracy 71.3% at epoch 27. This is the
strongest dropout 0.5 run of the three. Final gap was negative 2.3 points,
val still ahead of train.

## Interpretation
With three runs each now recorded: dropout 0.25 averaged 69.8% final val
accuracy and dropout 0.5 averaged 67.3%, about a 2.5 point difference. But
the spread within each setting is bigger than that: dropout 0.5 alone
ranged from 63.6% to 70.2%, a 6.6 point swing just from random luck between
runs. That is larger than the actual gap between the two settings' averages.
With only 3 runs each this is not enough data to say for sure that dropout
0.25 is really better and not just luck. It looks like a mild edge, not a
proven one.

## Next experiment
Run several more repeats of both settings, enough to get a real average
with less noise, before deciding anything for sure. Also worth trying
weight decay as a different kind of regularization while we are at it.
