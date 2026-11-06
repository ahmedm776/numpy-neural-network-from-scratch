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
