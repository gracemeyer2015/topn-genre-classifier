# Experiment notes

## Hypothesis
Replace flatten with global average pooling (shrinks the model from ~187k to
~24k parameters) while keeping the same dropout (p=0.5) from the last
experiment, to actually shrink the oversized final layer instead of just
making it harder to use.

## Result summary
train_acc and val_acc stay close together the whole run (57.3% vs 63.3% at
epoch 15, val actually ahead of train). Both train_loss and val_loss are
still steadily decreasing at epoch 15 with no sign of turning around (see
curves.png). Final val_acc (63.3%) is lower than the dropout-only run's
(72.5%), but this run hadn't finished improving yet.

## Interpretation
The overfitting gap is completely gone. Train and val track each other
instead of splitting apart. So GAP fixed the actual problem. But it's now
learning slowly and hadn't converged in 15 epochs, so it landed lower than
dropout-only did. Guessing dropout=0.5 is too strong now: dropping half of a
huge 16,384-value flatten still leaves plenty of signal, but dropping half of
a compact 64-value GAP output is a much bigger relative cut. That extra
regularization on top of a model that already lost ~87% of its parameters
may be more than it needs.

## Next experiment
Try dropout_rate=0 (no dropout at all) with the same GAP architecture, to see
if GAP alone is already enough to prevent overfitting. If train/val stay
aligned with no dropout, dropout isn't needed anymore. If the gap reopens,
dropout is still doing real work and picking a rate becomes a real tuning
question instead of a guess.
