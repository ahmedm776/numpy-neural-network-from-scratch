"""
Full official MNIST test-set evaluation of the frozen NumPy from-scratch model.

INFERENCE ONLY: loads the frozen weights (mnist_frozen_weights_repro.npz) and
scores them on the official 10,000-image MNIST test set. No training, no tuning.

Data: canonical MNIST IDX files (train/test), MD5-verified against the
checksums pinned in torchvision's torchvision/datasets/mnist.py
(f68b3c2dcbeaaa9fbdd348bbdeb94873, d53e105ee54ea40749a09fcbcd1e9432,
9fb629c4189551a2d022fa330f9573f3, ec29112dd5afa0611ce80d1b7f02629c) and the
same values on Zenodo record 10058130. Mirror used:
https://ossci-datasets.s3.amazonaws.com/mnist/

Preprocessing: reshape to 784, scale by 1/255.0 (exactly as the notebook).

Provenance: the frozen weights shipped alongside this script were instantiated
by re-running the notebook's deterministic training procedure exactly
(np.random.seed(42), fixed hyperparameters, fixed 100 epochs). Matching results across independent runs
(Iris 1.0; test[0:5700] accuracy 0.958421052631579) are consistent with an
identical model but do not prove bit-identical weights.
"""
import gzip, struct, sys
import numpy as np

def read_idx(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
    assert data[0] == 0 and data[1] == 0 and data[2] == 0x08
    ndim = data[3]
    dims = struct.unpack('>' + 'I' * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data[4 + 4 * ndim:], dtype=np.uint8).reshape(dims)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    a1 = sigmoid(X @ W1 + b1)
    return softmax(a1 @ W2 + b2)

def main(data_dir='.'):
    x_test = read_idx(f'{data_dir}/t10k-images-idx3-ubyte.gz')
    y_test = read_idx(f'{data_dir}/t10k-labels-idx1-ubyte.gz')
    X = x_test.reshape(x_test.shape[0], 784) / 255.0
    w = np.load('mnist_frozen_weights_repro.npz')
    pred = np.argmax(forward(X, w['W1'], w['b1'], w['W2'], w['b2']), axis=1)

    def acc(sl):
        a = float(np.mean(pred[sl] == y_test[sl]))
        return a, int(np.sum(pred[sl] == y_test[sl])), len(y_test[sl])

    a_head = acc(slice(0, 5700))     # the slice the notebook reported
    a_tail = acc(slice(5700, None))  # indices not previously reported
    a_full = acc(slice(None))
    print(f'test[0:5700]    : {a_head[0]:.15f} ({a_head[1]}/{a_head[2]})')
    print(f'test[5700:10000]: {a_tail[0]:.15f} ({a_tail[1]}/{a_tail[2]})')
    print(f'full test 10k   : {a_full[0]:.6f} ({a_full[1]}/{a_full[2]})')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
