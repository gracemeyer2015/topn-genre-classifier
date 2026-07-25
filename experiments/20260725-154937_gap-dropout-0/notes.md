# Experiment notes

## Hypothesis
Use the same GAP architecture as last time but eliminate dopout.
See if GAP alone is enough to stop overfitting, or if dropout was still doing
real work.

## Result summary
This run learns much faster than the dropout 0.5 version. By epoch 15 train
accuracy is 71.4% and val accuracy is 67.0%. Val loss drops steadily and
flattens out around 1.06 to 1.08 for the last few epochs. Train accuracy
starts out behind val accuracy early on. Around epoch 9 train pulls ahead of
val and the gap slowly grows to about 4 points by epoch 15.

## Interpretation
Turning off dropout made the model learn a lot faster. Final numbers are
better than the dropout 0.5 run in every way. But there is a small early
sign of overfitting starting near the end. Train is pulling ahead of val a
little and val loss stopped improving instead of still climbing down. It is
much milder than the original overfitting we saw before GAP. This tells me
dropout 0.5 was too strong. Dropout 0 might be slightly too weak if we train
longer. Somewhere in between is probably the right amount.

## Next experiment
Split dropout rate difference to 0.25. Keep GAP. Goal is to keep the fast 
learning we saw here while stopping that small late gap from growing if we
train for more epochs.
