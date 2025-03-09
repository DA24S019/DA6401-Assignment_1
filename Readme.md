Neural Network & Deep Learning for MNIST and Fashion-MNIST
This repository contains an implementation of a feedforward neural network used to train and test on the MNIST and Fashion-MNIST datasets. The project covers the essential aspects of neural network training—including forward propagation, back propagation, evaluation, and prediction—as well as hyperparameter tuning using Weights & Biases (W&B) sweeps.

Repository Links
W&B Repository: https://api.wandb.ai/links/da24s019-indian-institute-of-technology-madras/st30khic
GitHub Report: https://github.com/DA24S019/DA6401-Assignment_1.git
File Structure
feedforward_neural_network.py
Implements the Neural Network class which includes:

train: Trains the neural network.
forward_propagation: Implements the forward propagation step.
back_propagation: Implements the back propagation step.
evaluate: Computes validation accuracy and loss.
predict: Predicts output for given input samples.
activavtion_functions.py
Contains the class of activation functions used within the neural network.

optimizers.py
Implements all six optimizers used for network training.

dataset_loader.py
Loads the Fashion-MNIST or MNIST dataset and returns the training, validation, and testing sets.

arguments.py
Specifies the code parameters as outlined in the assignment.

category_logging.py
Logs the first occurrence of each of the 10 categories in the dataset as required by Question 1.

sweep_config.py
Contains the hyperparameters for network tuning and is configured for use with W&B sweep.

train_sweep.py
Implements hyperparameter tuning for the network using the W&B sweep function.

best_hyperparameters.py
After tuning, this file uses the best hyperparameters to generate a confusion matrix for the test data.

Getting Started
Prerequisites
Python 3.x
Required libraries (install via pip):
numpy
matplotlib
wandb

