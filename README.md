# Neural Network From Scratch vs Keras (DSAI253 Machine Learning project)

Recovered university coursework, packaged for Ahmed Khalifa's portfolio. A neural
network implemented twice and compared: once from scratch in pure NumPy (forward
propagation, backpropagation, softmax cross-entropy, gradient descent all written
by hand) and once with Keras, on two datasets (Iris and MNIST).

From the coursework archive of Ahmed Khalifa (DSAI253 Machine Learning).
The team/solo context of the original submission is not recorded, so no
sole-authorship claim is made for it. Packaged for portfolio publication from
the preserved original notebook with AI-assistant help.

## Verified results (re-run end-to-end on Google Colab, 2026-11-04)

| Dataset | Model | Accuracy | Final loss |
|---|---|---|---|
| Iris (2-class subset) | From scratch (NumPy) | 1.0000 | 0.0032 |
| Iris | Keras | 1.0000 | 0.1143 |
| MNIST (19,700 train / 5,700 test subset) | From scratch (NumPy) | 0.9584 | 0.0122 |
| MNIST (same subset) | Keras | 0.9326 | 0.2249 |

Note: the notebook deliberately uses a reduced MNIST subset (19,700 train / 5,700
test images), not the full 60k/10k split, so these numbers are not comparable to
published full-MNIST benchmarks. Iris is used as a binary (classes 0 and 1) task
for the scratch model.

Reproduction note (verified 2026-11-04, second independent run): the NumPy scratch
results are bit-exact across runs (MNIST scratch 0.958421052631579, Iris 1.0). The
Keras MNIST figure is NOT exactly reproducible: the notebook does not seed
TensorFlow, and three observed runs (Google Colab at the original verification:
0.9326; local tensorflow-cpu 2.21, 2026-11-04: 0.9274 and 0.9303) fell in the
0.9274-0.9326 range. The absent seed is a plausible cause, not a proven sole cause
(environment/version differences also apply); this range is an observation from
three runs, not a guaranteed band. A future deterministic variant could seed the
relevant RNGs, but the historical numbers above remain as originally observed.

## Fix applied during recovery (documented, not hidden)

The preserved original notebook crashed at the MNIST section with
`NameError: name 'y_train_onehot' is not defined` - the one-hot encoding step was
missing. One line was added in front of that cell, marked with a code comment:

    y_train_onehot = np.eye(10)[y_train_mnist]

Nothing else was changed. Every other cell matches the original except that
non-breaking spaces (U+00A0) in the original file were normalized to regular
spaces - a whitespace-only change with no effect on the code. The packaged
notebook carries no saved cell outputs; the executed-run evidence is
verification_log.md plus the comparison-table screenshot held with it. Code in
this file is identical to what actually ran during verification. See
verification_log.md for the full before/after evidence.

## What is in this folder

- ml_numpy_vs_keras.ipynb - the full notebook (fixed version), code and markdown
  only (no saved outputs); runs top-to-bottom on Colab with no data downloads
  other than keras.datasets MNIST (auto).
- mnist_frozen_weights_repro.npz - frozen weights of the NumPy from-scratch
  MNIST model, instantiated by re-running the notebook's exact deterministic
  training procedure (np.random.seed(42), fixed hyperparameters, fixed 100
  epochs). Lets anyone score the model without retraining.
- mnist_full_test_eval.py - inference-only scorer: loads the frozen weights and
  evaluates on the full official 10,000-image MNIST test set (canonical IDX
  files, MD5-pinned). Reproduces the recorded numbers bit-exactly.
- FULL_TEST_EVALUATION.md - the full-test evaluation record (2026-11-05).
- verification_log.md - what was run on 2026-11-04, the original failure, the fix,
  and the final metrics.
- requirements.txt - Python dependencies.
- linkedin_description.md - honest short description drafted for LinkedIn.
- addendum/ - matched-protocol 30-run experiment (2026-11-05) separating
  framework from training recipe; see addendum/README.md.

## Original preserved artifact

The original notebook is preserved unchanged in the owner's private archive;
this repo carries the packaged source copy.

## How to run

Open ml_numpy_vs_keras.ipynb in Google Colab, Runtime > Run all. No Drive mount,
no data files, no API keys needed. Full run takes about 2-3 minutes on a free
CPU runtime. The scratch NumPy MNIST training (100 epochs, 19,700 images) is the
slowest part.
