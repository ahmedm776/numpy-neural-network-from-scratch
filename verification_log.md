# DSAI253 ML project - verification log (2026-11-04, ~06:03-06:12 EET)

Verifier: Instinct agent. Environment: Google Colab free CPU runtime (Python 3).
Method: opened the preserved original notebook, ran "Run all" unchanged first
to establish the baseline, then applied the minimal fix in a scratch COPY and
re-ran end to end. The original notebook file was never modified.

## Step 1 - original notebook as found: FAILS

Run all on the untouched original stopped at the MNIST section:

    NameError: name 'y_train_onehot' is not defined
    ---> print("y_train_onehot shape:", y_train_onehot.shape)

The one-hot encoding step was missing from the notebook; the variable is used
again as the training target in the next cell, so the MNIST half of the project
could never have run in this state.

## Step 2 - minimal fix in a scratch copy

Added one line in front of that cell (marked with a code comment):

    y_train_onehot = np.eye(10)[y_train_mnist]

No other cell was changed.

## Step 3 - fixed notebook: full run passes

Run all completed with no errors. Key outputs:

    Iris from scratch:  Accuracy: 1.0
    MNIST from scratch: Accuracy: 0.958421052631579  (100 epochs)
    Keras Iris:         Accuracy: 1.0
    Keras MNIST:        Accuracy: 0.9326470494270325, Loss: 0.22493866086006165

Final comparison table rendered by the notebook:

    Dataset  Model         Accuracy   Loss
    IRIS     From Scratch  1.000000   0.003150
    IRIS     Keras         1.000000   0.114345
    MNIST    From Scratch  0.958421   0.012207
    MNIST    Keras         0.932647   0.224939

Caveats: MNIST is a 19,700/5,700 subset chosen by the original author, not the
full benchmark split; Iris from-scratch is a 2-class subset. The scratch copy
used for verification was deleted after the run.

Artifact note: the packaged ml_numpy_vs_keras.ipynb contains code and markdown
only - no cell outputs were persisted from this run. Its code is identical to
what actually ran (the original had non-breaking spaces throughout; both the
executed copy and the packaged file have them normalized to regular spaces, a
whitespace-only difference). Evidence of the executed outputs is the
comparison-table screenshot taken during the run (held by the verifier).
