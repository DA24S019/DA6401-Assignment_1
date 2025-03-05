import numpy as np
from Backpropagation_Algorithms import SGD, Momentum, NAG, RMSprop, Adam, Nadam

''' Class for activation functions and their derivatives '''
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
    def __init__(self, layer_sizes, activation="sigmoid", init_method="random", 
                 learning_rate=0.1, weight_decay=0, batch_size=32, optimizer="sgd"):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2 Regularization
        self.batch_size = batch_size
        self.wab = WeightAndBiasInitializer(layer_sizes, init_method)
        
        # Optimizer Selection
        optimizers = {
            "sgd": SGD(learning_rate),
            "momentum": Momentum(learning_rate),
            "nesterov": NAG(learning_rate),
            "rmsprop": RMSprop(learning_rate),
            "adam": Adam(learning_rate),
            "nadam": Nadam(learning_rate)
        }
        self.optimizer = optimizers.get(optimizer, SGD(learning_rate))  # Default to SGD

        # Activation function selection
        activation_functions = {
            "sigmoid": ActivationFunction.sigmoid,
            "tanh": ActivationFunction.tanh,
            "relu": ActivationFunction.relu
        }
        if activation not in activation_functions:
            raise ValueError(f"Unsupported activation function '{activation}'. Choose from {list(activation_functions.keys())}.")
        self.activation_func = activation_functions[activation]

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

    ''' Backward Propagation with L2 Regularization '''
    def backwardProp(self, X, Y):
        m = X.shape[0]
        dA = self.A[-1] - Y
        dW, db = [], []

        for i in reversed(range(len(self.wab.weights))):
            dZ = dA if i == len(self.wab.weights) - 1 else dA * self.activation_func(self.Z[i], derivative=True)
            dW_i = np.dot(self.A[i].T, dZ) / m
            db_i = np.sum(dZ, axis=0, keepdims=True) / m

            # Apply L2 Regularization
            dW_i += (self.weight_decay / m) * self.wab.weights[i]

            dW.insert(0, dW_i)
            db.insert(0, db_i)
            dA = np.dot(dZ, self.wab.weights[i].T)

        self.optimizer.update(self.wab.weights, self.wab.biases, dW, db)

    ''' Train the network with Mini-batch Gradient Descent '''
    def train(self, X, Y, epochs=10, batch_size=32):
        m = X.shape[0]  # Total number of training samples

        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)  # Shuffle training data each epoch
    
            total_loss = 0
            correct_predictions = 0

            for i in range(0, m, batch_size):
                batch_indices = indices[i : i + batch_size]
                X_batch = X[batch_indices]  # Select batch from input
                Y_batch = Y[batch_indices]  # Select batch from labels

                output = self.forwardProp(X_batch)  # Forward pass
                self.backwardProp(X_batch, Y_batch)  # Backpropagation
        
                batch_loss = -np.mean(Y_batch * np.log(output + 1e-8))  # Compute batch loss
                total_loss += batch_loss * len(X_batch)

                predicted_labels = np.argmax(output, axis=1)
                true_labels = np.argmax(Y_batch, axis=1)
                correct_predictions += np.sum(predicted_labels == true_labels)

            avg_loss = total_loss / m
            accuracy = correct_predictions / m

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, accuracy

    ''' Predict class labels '''
    def predict(self, X):
        output = self.forwardProp(X)
        return np.argmax(output, axis=1)  # Return class index with max probability
    

    def compute_loss(self, X, Y):
        """Compute cross-entropy loss"""
        output = self.forwardProp(X)
        loss = -np.mean(Y * np.log(output + 1e-8))  # Cross-entropy loss
        return loss
    

    def evaluate(self, X, Y):
        output = self.forwardProp(X)
        loss = -np.mean(Y * np.log(output + 1e-8))
        acc = np.mean(np.argmax(output, axis=1) == np.argmax(Y, axis=1))
        return loss, acc
    
    def compute_accuracy(self, X, Y):
        """
        Computes accuracy by comparing predicted labels with true labels.
        """
        output = self.forwardProp(X)  # Forward pass to get predictions
        predicted_labels = np.argmax(output, axis=1)
        true_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy
    

    def compute_loss(self, X, Y):
        """
        Computes the loss using cross-entropy.
        """
        output = self.forwardProp(X)
        loss = -np.mean(Y * np.log(output + 1e-8))
        return loss


