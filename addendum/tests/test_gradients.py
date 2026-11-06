"""Finite-difference check of the notebook's hand-written backprop (MNIST softmax net and Iris sigmoid net).
The notebook updates weights inline, so the analytic gradient is recovered from one SGD step with lr=1:
grad = (W_before - W_after)."""
import copy
import numpy as np
from scratch_nn import NeuralNetwork, NeuralNetworkMNIST

def num_grad(loss_fn, W, eps=1e-5):
    g = np.zeros_like(W)
    it = np.nditer(W, flags=["multi_index"])
    for _ in it:
        i = it.multi_index; old = W[i]
        W[i] = old + eps; lp = loss_fn()
        W[i] = old - eps; lm = loss_fn()
        W[i] = old; g[i] = (lp - lm) / (2 * eps)
    return g

def test_mnist_softmax_net():
    rng = np.random.RandomState(0)
    X = rng.rand(12, 6); y = np.eye(3)[rng.randint(0, 3, 12)]
    net = NeuralNetworkMNIST(6, 5, 3, 1.0); net.W1 = rng.randn(6, 5) * .5; net.W2 = rng.randn(5, 3) * .5; net.b1 = rng.randn(1, 5) * .1; net.b2 = rng.randn(1, 3) * .1
    ref = copy.deepcopy(net)
    def loss():
        p = ref.forward(X); return -np.mean(np.sum(y * np.log(p), axis=1))
    net.train(X, y, epochs=1, batch_size=12)  # one full-batch step, lr=1
    for name in ("W1", "W2", "b1", "b2"):
        analytic = getattr(ref, name) - getattr(net, name)
        numeric = num_grad(loss, getattr(ref, name))
        assert np.allclose(analytic, numeric, atol=1e-6), name

def test_iris_sigmoid_net():
    rng = np.random.RandomState(1)
    X = rng.randn(10, 4); y = rng.randint(0, 2, 10).astype(float)
    net = NeuralNetwork(4, 3, 1, 1.0); net.W1 = rng.randn(4, 3) * .5; net.W2 = rng.randn(3, 1) * .5; net.b1 = rng.randn(1, 3) * .1; net.b2 = rng.randn(1, 1) * .1
    ref = copy.deepcopy(net)
    def loss():
        p = ref.forward(X); yy = y.reshape(-1, 1)
        return -np.mean(yy * np.log(p + 1e-8) + (1 - yy) * np.log(1 - p + 1e-8))
    net.train(X, y, epochs=1)
    for name in ("W1", "W2", "b1", "b2"):
        analytic = getattr(ref, name) - getattr(net, name)
        numeric = num_grad(loss, getattr(ref, name))
        assert np.allclose(analytic, numeric, atol=1e-6), name
