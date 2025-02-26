import numpy as np

class SGD:
    """Stochastic Gradient Descent (SGD) optimizer."""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weights, biases, dW, db):
        for i in range(len(weights)):
            weights[i] -= self.lr * dW[i]
            biases[i] -= self.lr * db[i]

class Momentum:
    """Momentum-based Gradient Descent optimizer."""
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biaseSs, dW, db):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.velocity_w[i] = self.beta * self.velocity_w[i] + (1 - self.beta) * dW[i]
            self.velocity_b[i] = self.beta * self.velocity_b[i] + (1 - self.beta) * db[i]
            weights[i] -= self.lr * self.velocity_w[i]
            biases[i] -= self.lr * self.velocity_b[i]

class NAG:
    """Nesterov Accelerated Gradient (NAG) optimizer."""
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, dW, db):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            lookahead_w = weights[i] - self.beta * self.velocity_w[i]
            lookahead_b = biases[i] - self.beta * self.velocity_b[i]

            self.velocity_w[i] = self.beta * self.velocity_w[i] + self.lr * dW[i]
            self.velocity_b[i] = self.beta * self.velocity_b[i] + self.lr * db[i]

            weights[i] = lookahead_w - self.lr * dW[i]
            biases[i] = lookahead_b - self.lr * db[i]
