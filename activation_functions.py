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
    
    ''' Identity function '''
    @staticmethod
    def identity(val_arr , derivative=False):
        if derivative==True:
            return 1
        return val_arr
