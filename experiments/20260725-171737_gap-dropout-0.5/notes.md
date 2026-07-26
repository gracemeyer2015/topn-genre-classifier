# Experiment notes

## Hypothesis
Run dropout 0.5 for 30 epochs instead of 15. At 15 epochs it was the
slowest of the three and had not finished improving. See if given more time
it catches up to the others.

## Result summary
Train accuracy ends at 64.6% and val accuracy ends at 68.0%. Val accuracy is
still ahead of train accuracy the entire run, gap is negative 3.4 points at
epoch 30. Best val accuracy during the run was 69.2% at epoch 29. Val loss
trends down the whole time with some noise, best point is 1.03 at epoch 29.

## Interpretation
Even after 30 epochs this is still the most cautious of the three, val
never falls behind train at any point. Its final val accuracy (68.0%) is
close to the other two configs, so it did mostly catch up, but it took the
whole 30 epochs to get there while dropout 0.25 reached a similar or better
point faster and dropout 0 reached it even earlier at the cost of
overfitting. Dropout 0.5 is safe but not the most efficient choice.

## Next experiment
None needed for this value. Comparing all three at 30 epochs together,
dropout 0.25 looks like the best overall balance: nearly identical final val
accuracy to the other two, small and stable train val gap, and reasonably
fast convergence. Recommend locking in dropout 0.25 with the GAP
architecture and moving to saving a checkpoint for handoff.
