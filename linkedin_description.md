Machine Learning course project (DSAI253): built a neural network twice - once
from scratch in pure NumPy (forward pass, backpropagation, softmax cross-entropy
and gradient descent implemented by hand) and once in Keras - and compared them
on Iris and MNIST. The NumPy model reached 95.8% on an MNIST subset vs 93.3% for
the Keras baseline, and 100% on the Iris binary task for both. Packaged with a
one-line fix for a missing label-encoding step so the notebook runs end-to-end;
full verification log included.
