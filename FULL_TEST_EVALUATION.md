# Full official MNIST test-set evaluation (2026-11-05)

The shipped notebook reports the NumPy from-scratch model on the first 5,700
images of the official MNIST test set (0.958421052631579 = 5463/5700). This
note records an exact-procedure reproduction run scored on the full official
10,000-image test set.

## What was run

The package now ships the frozen weights (`mnist_frozen_weights_repro.npz`).
They were instantiated by re-running the notebook's deterministic training
procedure exactly (np.random.seed(42), fixed hyperparameters, fixed 100
epochs, no early stopping; same seeds, draw order, hyperparameters), and were
then scored on the full test set (`mnist_full_test_eval.py`, inference only).

Matching results across the independent runs (Iris scratch 1.0; test[0:5700]
accuracy 0.958421052631579, bit-equal) are consistent with an identical model
but do not prove bit-identical weights; floating-point/BLAS drift between
environments is possible.

## Results (this reproduction run)

| Slice | Accuracy | Correct |
| --- | --- | --- |
| test[0:5700] (the slice the notebook reported) | 0.958421052631579 | 5463/5700 |
| test[5700:10000] (indices not previously reported) | 0.9739534883720931 | 4188/4300 |
| full official test (10,000) | 0.9651 | 9651/10000 |

## Data identity

Canonical MNIST IDX files from https://ossci-datasets.s3.amazonaws.com/mnist/,
MD5-verified against the checksums pinned in torchvision's
torchvision/datasets/mnist.py (train images f68b3c2dcbeaaa9fbdd348bbdeb94873,
train labels d53e105ee54ea40749a09fcbcd1e9432, test images
9fb629c4189551a2d022fa330f9573f3, test labels
ec29112dd5afa0611ce80d1b7f02629c); the same checksums are listed on Zenodo
record 10058130. (The notebook's original loader source,
storage.googleapis.com/cvdf-datasets/mnist/mnist.npz, no longer serves the
file.)

## Limits

- Fitting used only train[0:19700]; no test images were used in fitting, and
  training ran a fixed 100 epochs with no validation-based stopping.
- test[0:5700] was the development-reported metric in the original work, so
  its exposure to the original development process is known. The original
  development exposure of any other test index (5700-9999) is unknown.
- The full-10k accuracy above is therefore NOT independent confirmation of
  generalization; it mixes the reported slice with previously unreported
  indices. No conclusion about selection bias in either direction is drawn.
- Preprocessing is reshape + /255.0 exactly as the notebook.
