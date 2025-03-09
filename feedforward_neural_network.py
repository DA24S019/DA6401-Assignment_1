import numpy as np
from optimizers import SGD, Momentum, NAG, RMSprop, Adam, Nadam
from activation_functions import ActivationFunction
from arguments import get_args
''' Class for activation functions and their derivatives '''


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
            input_dimension, output_dimension = self.layer_sizes[i], self.layer_sizes[i + 1]

            # Weight Initialization
            if self.init_method == "random":
                limit=1
                W=np.random.uniform(-limit, limit, size=(input_dimension, output_dimension))
                #W = np.random.randn(input_dim, output_dim) * 0.01  # Small random values
            elif self.init_method == "xavier":
                limit = np.sqrt(6 / (input_dimension + output_dimension))
                W=np.random.uniform(-limit, limit, size=(input_dimension, output_dimension))
                #W = np.random.randn(input_dimension, output_dimension) * np.sqrt(1.0 / input_dimension)

            self.weights.append(W)
            self.biases.append(np.zeros((1, output_dimension)))


''' Feedforward Neural Network Implementation '''
class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activation, init_method, 
                 learning_rate, weight_decay, batch_size, optimizer):
        args=get_args()
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # L2 Regularization
        self.batch_size = batch_size
        self.wab = WeightAndBiasInitializer(layer_sizes, init_method)
        self.momentum = args.momentum
        self.epsilon = args.epsilon
        self.beta1 =args.beta1
        self.beta2 =args.beta2
        self.loss_function=args.loss


        # Optimizer Selection
        optimizers = {
            "sgd": SGD(self.learning_rate ),
            "momentum": Momentum(self.learning_rate ,self.momentum),
            "nesterov": NAG(self.learning_rate ,self.momentum),
            "rmsprop": RMSprop(self.learning_rate ,self.weight_decay,self.epsilon),
            "adam": Adam(self.learning_rate,self.beta1,self.beta2, self.epsilon),
            "nadam": Nadam(self.learning_rate,self.beta1,self.beta2, self.epsilon )
        }
        self.optimizer = optimizers[optimizer]  # Default to SGD

        # Activation function selection
        activation_functions = {
            "sigmoid": ActivationFunction.sigmoid,
            "tanh": ActivationFunction.tanh,
            "relu": ActivationFunction.relu,
            "identity": ActivationFunction.identity
        }
        
        self.activation_func = activation_functions[activation]

    ''' Forward Propagation '''
    def forwardProp(self, X):
        self.post_activation = [X]  # Input layer
        self.pre_activation = []   # Store pre-activation values

        for i, (W, b) in enumerate(zip(self.wab.weights, self.wab.biases)):
            Z = np.dot(self.post_activation[-1], W) + b  # Linear transformation
            if i == len(self.wab.weights) - 1:
                A = ActivationFunction.softmax(Z)  # Softmax for last layer
            else:
                A = self.activation_func(Z)  # Chosen activation for hidden layers

            self.pre_activation.append(Z)
            self.post_activation.append(A)

        return self.post_activation[-1]  # Output activations

    ''' Backward Propagation'''
    def backwardProp(self, X, Y):
        m = X.shape[0]
        dW, db = [], []

        if self.loss_function == 'cross_entropy':
            dA = self.post_activation[-1] - Y
        elif self.loss_function == 'mean_squared_error':
            dA = (self.post_activation[-1] - Y) * self.post_activation[-1] * (1 - self.post_activation[-1])   
        

        for i in reversed(range(len(self.wab.weights))):
            dZ = dA if i == len(self.wab.weights) - 1 else dA * self.activation_func(self.pre_activation[i], derivative=True)
            dW_i = np.dot(self.post_activation[i].T, dZ) / m
            db_i = np.sum(dZ, axis=0, keepdims=True) / m

            # Apply L2 Regularization
            dW_i += (self.weight_decay / m) * self.wab.weights[i]

            dW.insert(0, dW_i)
            db.insert(0, db_i)
            dA = np.dot(dZ, self.wab.weights[i].T)

        self.optimizer.update(self.wab.weights, self.wab.biases, dW, db)

    ''' Train the network with Mini-batch Gradient Descent '''
    def train(self, X, Y, epochs, batch_size):
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
    


