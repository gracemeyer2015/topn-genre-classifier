# Tensor Contract

**Pipeline -> Model input:** `(N, 1, 128, 130)`, float32
- N = batch size
- 1 = single channel
- 128 = mel frequency bands
- 130 = time frames per 3-second clip

**Model output:** `(N, 10)`, raw logits (no softmax - CLI applies that later)

**Labels:** genre index (0-9):

| Index | Genre | Index | Genre |
|---|---|---|---|
| 0 | blues | 5 | jazz |
| 1 | classical | 6 | metal |
| 2 | country | 7 | pop |
| 3 | disco | 8 | reggae |
| 4 | hiphop | 9 | rock |
