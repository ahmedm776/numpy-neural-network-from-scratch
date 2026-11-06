"""Matched-protocol comparison. usage: run_arm.py ARM SEED   (ARM: scratch | keras_sgd | keras_adam)
Protocol fixed BEFORE any run (see README): fit=train[0:17700], val=train[17700:19700], test=official 10k (scored once per run, no selection on it)."""
import sys, os, io, json, time, contextlib
import numpy as np
from mnist_io import load
arm, seed = sys.argv[1], int(sys.argv[2])
out = f"results/{arm}_seed{seed}.json"
if os.path.exists(out): sys.exit(0)
xtr, ytr, xte, yte = load()
X = xtr[:19700] / 255.0; Y = ytr[:19700]
Xf, Yf, Xv, Yv = X[:17700], Y[:17700], X[17700:], Y[17700:]
Xt = xte / 255.0
t0 = time.time()
if arm == "scratch":
    from scratch_nn import NeuralNetworkMNIST
    np.random.seed(seed)
    m = NeuralNetworkMNIST(784, 64, 10, 0.5)
    with contextlib.redirect_stdout(io.StringIO()):
        m.train(Xf, np.eye(10)[Yf], epochs=100)
    pv, pt = m.predict(Xv), m.predict(Xt)
else:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf, keras
    keras.utils.set_random_seed(seed)
    init = keras.initializers.RandomNormal(stddev=0.01, seed=seed) if arm == "keras_sgd" else "glorot_uniform"
    m = keras.Sequential([keras.layers.Input(shape=(784,)), keras.layers.Dense(64, activation="sigmoid", kernel_initializer=init, bias_initializer="zeros"),
                          keras.layers.Dense(10, activation="softmax", kernel_initializer=init, bias_initializer="zeros")])
    if arm == "keras_sgd":
        m.compile(optimizer=keras.optimizers.SGD(0.5), loss="categorical_crossentropy"); ep, bs = 100, 128
    else:
        m.compile(optimizer="adam", loss="categorical_crossentropy"); ep, bs = 10, 32
    m.fit(Xf, np.eye(10)[Yf], epochs=ep, batch_size=bs, verbose=0)
    pv, pt = m.predict(Xv, verbose=0).argmax(1), m.predict(Xt, verbose=0).argmax(1)
r = dict(arm=arm, seed=seed, val_acc=float((pv == Yv).mean()), test_acc=float((pt == yte).mean()),
         test_0_5700=float((pt[:5700] == yte[:5700]).mean()), test_5700_10000=float((pt[5700:] == yte[5700:]).mean()), seconds=round(time.time() - t0, 1))
json.dump(r, open(out, "w")); print(r)
