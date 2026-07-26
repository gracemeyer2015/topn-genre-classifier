# Experiment notes

## Hypothesis
Run dropout 0 for 30 epochs instead of 15. At 15 epochs it already showed a
small overfitting gap starting around epoch 6 to 7. Training longer should
show whether that gap grows or settles down.

## Result summary
Train accuracy keeps climbing the whole run and ends at 79.6%. Val accuracy
ends at 68.5% but bounces around a lot in the second half, between about 59%
and 71%. The gap between train and val accuracy is 11.1 points by epoch 30,
up from about 4 points at epoch 15. Best val accuracy hit during the run was
71.3% at epoch 23, and best val loss was 0.97 at epoch 24, but neither point
held, val loss and val accuracy both got noisier and worse after that.

## Interpretation
The early warning sign from the 15 epoch run was real. With more training
the gap kept growing instead of settling down. The model keeps improving on
training data but is not getting any better on new songs, it might even be
getting slightly worse and less stable. This confirms dropout 0 is not a
safe long term choice, even though its early numbers looked good.

## Next experiment
Compare directly against dropout 0.25 and dropout 0.5 at the same 30 epochs.
If they reach similar val accuracy without this growing gap, that is the
better choice even though its raw numbers are not the single highest.
