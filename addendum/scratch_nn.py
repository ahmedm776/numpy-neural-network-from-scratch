"""Verbatim copies of the notebook cells 1-3 (helpers + the two from-scratch classes)."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def one_hot(y, num_classes):
    result = np.zeros((y.shape[0], num_classes))
    result[np.arange(y.shape[0]), y] = 1
    return result


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.losses = []
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    def train(self, X, y, epochs):
        y = y.reshape(-1, 1)
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = -np.mean(
                y * np.log(y_pred + 1e-8) +
                (1 - y) * np.log(1 - y_pred + 1e-8)
            )
            self.losses.append(loss)
            m = X.shape[0]
            dz2 = y_pred - y
            dW2 = np.dot(self.a1.T, dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m
            dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
            dW1 = np.dot(X.T, dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            if epoch % 500 == 0:
                print("Epoch:", epoch, "Loss:", loss)
    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


class NeuralNetworkMNIST:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.lr = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.losses = []
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2
    def train(self, X, y, epochs, batch_size=128):
        n = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                y_pred = self.forward(X_batch)
                m = X_batch.shape[0]
                dz2 = y_pred - y_batch
                dW2 = np.dot(self.a1.T, dz2) / m
                db2 = np.sum(dz2, axis=0, keepdims=True) / m
                dz1 = np.dot(dz2, self.W2.T) * self.a1 * (1 - self.a1)
                dW1 = np.dot(X_batch.T, dz1) / m
                db1 = np.sum(dz1, axis=0, keepdims=True) / m
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            train_pred = self.forward(X)
            loss = -np.mean(np.sum(y * np.log(train_pred + 1e-8), axis=1))
            self.losses.append(loss)
            if epoch % 5 == 0:
                print("Epoch:", epoch, "Loss:", loss)
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)


