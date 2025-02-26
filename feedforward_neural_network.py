import numpy as np

''' Class for activation functions and their derivatives '''
import numpy as np

class ActivationFunction:
    ''' Sigmoid function '''
    @staticmethod
    def sigmoid(x, derivative=False):
        x = np.array(x)  
        sig = 1 / (1 + np.exp(-x))  
        return sig * (1 - sig) if derivative else sig

    ''' Tanh function '''
    @staticmethod
    def tanh(x, derivative=False):
        x = np.array(x)
        th = np.tanh(x)  
        return 1 - th**2 if derivative else th  

    ''' ReLU function '''
    @staticmethod
    def relu(x, derivative=False):
        x = np.array(x)  
        return np.where(x > 0, 1.0, 0.0) if derivative else np.maximum(0, x)  

    ''' Softmax function '''
    @staticmethod
    def softmax(x, derivative=False):
        x = np.array(x)
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
        softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)  

        if derivative:
            return softmax_x * (1 - softmax_x)  # Incorrect for full derivative, handled differently
        return softmax_x  



''' Initialize weights and biases '''
class WeightAndBiasInitializer:
    def __init__(self, layer_sizes, init_method="random"):
        self.layer_sizes = layer_sizes
        self.init_method = init_method
        self.weights = []
        self.biases = []
        self._initialize_weights()

    ''' Function to initialize weights '''
    def _initialize_weights(self):
        np.random.seed(42)

        for i in range(len(self.layer_sizes) - 1):
            input_dim, output_dim = self.layer_sizes[i], self.layer_sizes[i + 1]

            # Weight Initialization
            if self.init_method == "random":
                W = np.random.randn(input_dim, output_dim) * 0.01  # Small random values
            elif self.init_method == "xavier":
                W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)

            self.weights.append(W)
            self.biases.append(np.zeros((1, output_dim)))


''' Feedforward Neural Network Implementation '''
class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activation="sigmoid", init_method="random", learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activations = ActivationFunction()
        self.wab = WeightAndBiasInitializer(layer_sizes, init_method)

        # Ensure the activation function exists
        if hasattr(self.activations, activation):
            self.activation_func = getattr(self.activations, activation)
        else:
            raise ValueError(f"Activation function '{activation}' not found.")

    ''' Forward Propagation '''
    def forwardProp(self, X):
        self.A = [X]  # Input layer
        self.Z = []   # Store pre-activation values

        for i, (W, b) in enumerate(zip(self.wab.weights, self.wab.biases)):
            Z = np.dot(self.A[-1], W) + b  # Linear transformation
            if i == len(self.wab.weights) - 1:
                A = ActivationFunction.softmax(Z)  # Softmax for last layer
            else:
                A = self.activation_func(Z)  # Chosen activation for hidden layers

            self.Z.append(Z)
            self.A.append(A)

        return self.A[-1]  # Output activations
  # Output layer activation

    ''' Backward Propagation '''
    def backwardProp(self, X, Y):
        m = X.shape[0]
        dA = self.A[-1] - Y  # Softmax + Cross-Entropy Loss derivative

        for i in reversed(range(len(self.wab.weights))):
            if i == len(self.wab.weights) - 1:
                dZ = dA  # For softmax, dZ = dA
            else:
                dZ = dA * self.activation_func(self.Z[i], derivative=True)

            dW = np.dot(self.A[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.wab.weights[i].T)

            # Update weights
            self.wab.weights[i] -= self.learning_rate * dW
            self.wab.biases[i] -= self.learning_rate * db

    ''' Train the network '''
    def train(self, X, Y, epochs=10):
        for epoch in range(epochs):
            output = self.forwardProp(X)
            self.backwardProp(X, Y)
            loss = -np.mean(Y * np.log(output + 1e-8))  # Cross-entropy loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    ''' Predict class labels '''
    def predict(self, X):
        output = self.forwardProp(X)
        return np.argmax(output, axis=1)  # Return class index with max probability
