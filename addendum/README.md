# Addendum: matched-protocol check of "NumPy from scratch vs Keras" (2026-11-05)

Addendum to the ML project package (the original notebook, README and earlier verification files are unchanged).

## Why
The notebook's MNIST comparison (scratch 95.8% vs Keras 93.3%) compares different things: the scratch model trains on
19,700 images for 100 epochs with SGD (lr 0.5, batch 128, N(0,0.01) init) and is scored on test[0:5700]; the Keras model
trains on 15,200 images for 10 epochs with Adam (batch 32, default init) and is scored on test[0:3400]. So the notebook
cannot say whether the gap comes from the framework or from the training recipe. This addendum asks that question.

## Protocol (written before any run)
- Data: official MNIST, MD5-verified. Fit = train[0:17700], validation = train[17700:19700] (the notebook's 19,700 pool, split).
  Test = official 10,000 images, scored once per run. Nothing was selected, stopped or tuned on test; the three arms
  and hyperparameters were fixed before running. This is a multi-seed training experiment (30 runs, about 25 minutes of CPU). 10 seeds (0-9) per arm.
- Arms: (1) the notebook's NeuralNetworkMNIST class, verbatim (scratch_nn.py); (2) Keras, same 784-64(sigmoid)-10(softmax)
  net, SGD lr 0.5, batch 128, 100 epochs, N(0,0.01) init and zero biases (matched to scratch); (3) Keras with the notebook's
  optimizer recipe (Adam, batch 32, 10 epochs, default init) run on the NEW 17,700-image split. This is not the original
  shipped protocol (15,200 train / test[0:3400], unseeded); that original run is not reproduced here.

## Results

Spread = sample standard deviation (n-1) across 10 seeds. No confidence interval is computed.

| Arm | Seeds | Val acc % | Test acc % (official 10k) | test[0:5700] | test[5700:10000] |
|---|---|---|---|---|---|
| NumPy scratch (SGD 0.5, bs128, 100 ep) | 10 | 96.51 (sd 0.24) | 96.26 (sd 0.11) | 95.49 (sd 0.17) | 97.28 (sd 0.11) |
| Keras matched-config (SGD 0.5, bs128, 100 ep, N(0,0.01) init) | 10 | 96.47 (sd 0.31) | 96.25 (sd 0.11) | 95.44 (sd 0.13) | 97.32 (sd 0.14) |
| Keras Adam recipe (Adam, bs32, 10 ep) on the new 17,700 split | 10 | 95.47 (sd 0.11) | 94.78 (sd 0.13) | 93.60 (sd 0.18) | 96.35 (sd 0.16) |

Same-seed-index difference, scratch minus Keras-matched, 10k test (points): mean +0.013, sample sd 0.164, min -0.21, max +0.34. (Seed index is shared, but NumPy and Keras draw different random streams, so runs are not paired in any real sense.)
Adam recipe minus Keras-matched: mean -1.466, sample sd 0.191

Per-seed 10k test accuracy % (scratch / keras_sgd / keras_adam), val in brackets:
seed 0: 96.26 [96.65] / 96.26 [96.25] / 95.01 [95.35]
seed 1: 96.30 [96.65] / 96.41 [96.25] / 94.60 [95.35]
seed 2: 96.35 [96.65] / 96.21 [96.50] / 94.85 [95.65]
seed 3: 96.42 [96.45] / 96.08 [95.75] / 94.70 [95.60]
seed 4: 96.13 [96.10] / 96.22 [96.65] / 94.91 [95.45]
seed 5: 96.13 [96.75] / 96.34 [96.65] / 94.77 [95.40]
seed 6: 96.30 [96.40] / 96.24 [96.50] / 94.64 [95.35]
seed 7: 96.26 [96.80] / 96.39 [96.75] / 94.69 [95.55]
seed 8: 96.10 [96.15] / 96.12 [96.55] / 94.83 [95.50]
seed 9: 96.37 [96.50] / 96.22 [96.80] / 94.83 [95.55]

## Reading
Under this matched configuration the scratch-vs-Keras gap largely disappears: mean difference +0.01 points on the 10k test,
within the seed-to-seed spread (sd about 0.11-0.16 points). That is evidence about this one architecture and recipe, not a
proof that the frameworks are equivalent or that the recipe is the universal cause of the notebook's gap. The Adam recipe on the
new split is about 1.5 points lower than the matched configuration on the same 10k test, so optimizer/epochs matter here; amount of
training data and test slice are also different in the notebook and were not separated one by one. test[0:5700] scores about 2 points
lower than test[5700:10000] for every arm, so the reported slice alone moves accuracy by that much.
The notebook's own numbers (95.84% scratch on test[0:5700], 93.26% Keras on test[0:3400]) remain as originally observed.

## Implementation differences that remain between scratch and Keras-matched
- Seeds: scratch uses np.random.seed(seed) (initial weights and per-epoch shuffles); Keras uses keras.utils.set_random_seed(seed) plus a seeded
  initializer. Same seed index, different random streams, so initial weights and batch orders differ; the runs are not paired.
- Fitting data and order: identical images in identical file order (train[0:17700]); each epoch is reshuffled by each framework's own RNG. Last batch of each epoch has 76 samples in both.
- Loss: scratch adds 1e-8 inside log in the reported loss only (the update uses the analytic softmax-minus-label gradient); Keras clips probabilities at 1e-7. Training updates are plain mini-batch SGD without momentum in both.
- Initialization: scratch N(0,0.01) weights, zero biases; the matched arm uses RandomNormal(stddev 0.01) and zero biases. Float precision: scratch float64, Keras float32.
- Evaluation: final-epoch weights in all arms (no early stopping); validation is reported, not used to select anything.

## Gradient check: what it covers
tests/test_gradients.py takes ONE full-batch SGD step with lr=1 on tiny random networks (MNIST-class 6-5-3 softmax net with 12 samples; Iris-class 4-3-1 sigmoid
net with 10 samples), recovers the update's gradient, and compares all four parameter arrays (W1, W2, b1, b2) to central finite differences
(eps 1e-5) with numpy.allclose (atol 1e-6, default rtol 1e-5). It passes. It shows the notebook's gradient formulas match numerical derivatives for these
tiny random cases; it does not verify the multi-batch training loop, the full 784-64-10 size, the 1e-8 log epsilon's effect on reported loss, or anything beyond those formulas.

## Limits
- One architecture (64 hidden units), one learning rate; no tuning. 10 seeds, one environment (tensorflow-cpu 2.21.0, 2 CPU cores).
- test[0:5700] was the notebook's development-reported slice; this is not independent confirmation of generalization and makes no 95%-goal claim.

## Reproduce
`sh fetch_data.sh; python run_arm.py ARM SEED` for ARM in scratch|keras_sgd|keras_adam, SEED 0-9; `python summarize.py`; `python -m pytest tests`. Raw per-run results are in results/.
