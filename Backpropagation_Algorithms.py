import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, weights, biases, grad_w, grad_b):
        for i in range(len(weights)):
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]

class Momentum:
    def __init__(self, learning_rate=0.01, beta1=0.9):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.momentum_w = None
        self.momentum_b = None

    def update(self, weights, biases, grad_w, grad_b):
        # Initialize momentum terms if they are None
        if self.momentum_w is None:
            self.momentum_w = [np.zeros_like(w) for w in weights]
        if self.momentum_b is None:
            self.momentum_b = [np.zeros_like(b) for b in biases]

        # Apply momentum-based update
        for i in range(len(weights)):
            self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * grad_w[i]
            self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * grad_b[i]

            weights[i] -= self.learning_rate * self.momentum_w[i]
            biases[i] -= self.learning_rate * self.momentum_b[i]


class NAG:
    def __init__(self, learning_rate=0.01, momentum_coeff=0.9):
        self.lr = learning_rate
        self.momentum_coeff = momentum_coeff
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            lookahead_w = weights[i] - self.momentum_coeff * self.velocity_w[i]
            lookahead_b = biases[i] - self.momentum_coeff * self.velocity_b[i]

            self.velocity_w[i] = self.momentum_coeff * self.velocity_w[i] + self.lr * grad_w[i]
            self.velocity_b[i] = self.momentum_coeff * self.velocity_b[i] + self.lr * grad_b[i]

            weights[i] = lookahead_w - self.lr * grad_w[i]
            biases[i] = lookahead_b - self.lr * grad_b[i]

class RMSprop:
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.sq_grad_w = None
        self.sq_grad_b = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.sq_grad_w is None:
            self.sq_grad_w = [np.zeros_like(w) for w in weights]
            self.sq_grad_b = [np.zeros_like(b) for b in biases]

        for i in range(len(weights)):
            self.sq_grad_w[i] = self.decay_rate * self.sq_grad_w[i] + (1 - self.decay_rate) * grad_w[i]**2
            self.sq_grad_b[i] = self.decay_rate * self.sq_grad_b[i] + (1 - self.decay_rate) * grad_b[i]**2

            weights[i] -= self.lr * grad_w[i] / (np.sqrt(self.sq_grad_w[i]) + self.epsilon)
            biases[i] -= self.lr * grad_b[i] / (np.sqrt(self.sq_grad_b[i]) + self.epsilon)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.momentum_w = None
        self.momentum_b = None
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.momentum_w is None:
            self.momentum_w = [np.zeros_like(w) for w in weights]
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.momentum_b = [np.zeros_like(b) for b in biases]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        self.timestep += 1
        for i in range(len(weights)):
            self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * grad_w[i]
            self.velocity_w[i] = self.beta2 * self.velocity_w[i] + (1 - self.beta2) * (grad_w[i]**2)
            self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * grad_b[i]
            self.velocity_b[i] = self.beta2 * self.velocity_b[i] + (1 - self.beta2) * (grad_b[i]**2)

            m_w_hat = self.momentum_w[i] / (1 - self.beta1**self.timestep)
            v_w_hat = self.velocity_w[i] / (1 - self.beta2**self.timestep)
            m_b_hat = self.momentum_b[i] / (1 - self.beta1**self.timestep)
            v_b_hat = self.velocity_b[i] / (1 - self.beta2**self.timestep)

            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

class Nadam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.momentum_w = None
        self.momentum_b = None
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, grad_w, grad_b):
        if self.momentum_w is None:
            self.momentum_w = [np.zeros_like(w) for w in weights]
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.momentum_b = [np.zeros_like(b) for b in biases]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        self.timestep += 1
        for i in range(len(weights)):
            self.momentum_w[i] = self.beta1 * self.momentum_w[i] + (1 - self.beta1) * grad_w[i]
            self.velocity_w[i] = self.beta2 * self.velocity_w[i] + (1 - self.beta2) * (grad_w[i]**2)
            self.momentum_b[i] = self.beta1 * self.momentum_b[i] + (1 - self.beta1) * grad_b[i]
            self.velocity_b[i] = self.beta2 * self.velocity_b[i] + (1 - self.beta2) * (grad_b[i]**2)

            m_w_hat = (self.beta1 * self.momentum_w[i] + (1 - self.beta1) * grad_w[i]) / (1 - self.beta1**self.timestep)
            v_w_hat = self.velocity_w[i] / (1 - self.beta2**self.timestep)
            m_b_hat = (self.beta1 * self.momentum_b[i] + (1 - self.beta1) * grad_b[i]) / (1 - self.beta1**self.timestep)
            v_b_hat = self.velocity_b[i] / (1 - self.beta2**self.timestep)

            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
