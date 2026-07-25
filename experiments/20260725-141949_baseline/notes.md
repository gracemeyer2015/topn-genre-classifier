# Experiment notes

## Hypothesis
Run the plain model as-is (no dropout, no architecture changes) for 15 epochs
to see how well it does before trying any fixes.

## Result summary
Training accuracy climbs to almost 100% by epoch 7 and stays there. Validation
accuracy gets stuck around 69-72% the whole time and never really improves
past the first few epochs. Validation loss stops improving after epoch 3 and
then keeps getting worse for the rest of the run, even though training loss
keeps dropping the whole time (see curves.png).

## Interpretation
The model is memorizing the training songs instead of learning general genre
patterns. It has more room to memorize than it needs. Almost all of its
parameters are in the very last layer, which maps every pixel of the
processed audio directly to a genre guess with nothing forcing it to
generalize.

## Next experiment
Shrink that last layer down (global average pooling before it, instead of
flattening everything) and add dropout, so the model can't just memorize the
answers anymore. Compare the new validation curve to this one. Looking for
val_loss to keep improving further into training instead of turning upward
around epoch 3.
