import numpy as np
from arguments import get_args
from dataset_loader import load_data  
from  feedforward_neural_network import FeedforwardNeuralNetwork
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from arguments import get_args


def main():
    args=get_args()
    dataset=args.dataset

    X_train, Y_train, X_test, Y_test = load_data(dataset)

    activation = args.activation                  # Activation function for hidden layers
    optimizer = args.optimizer                  # Optimizer: "sgd", "momentum", "nag", "rmsprop", "adam", "nadam"
    first_layer = X_train.shape[1]  # Integer
    output_layer = Y_train.shape[1]  # Integer
    num_layers = args.num_layers  # Integer
    hidden_size = args.hidden_size  # Integer
    epochs=args.epochs
    learning_rate=args.learning_rate
    weight_decay=args.weight_decay
    batch_size=args.batch_size
    weight_init=args.weight_init
    # Construct the layers list correctly
    layer_sizes = [first_layer] + [hidden_size] * num_layers + [output_layer]
    #layer_sizes = [784,128,128,10]
    # Initialize the network using your FeedforwardNeuralNetwork class.
    nn = FeedforwardNeuralNetwork(layer_sizes, activation, weight_init, learning_rate, weight_decay, batch_size, optimizer)
    nn.train(X_train,Y_train,epochs,batch_size) 


    y_pred = nn.predict(X_test)
    y_true = np.argmax(Y_test, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define class names (for example, MNIST digits 0-9)
    if args.dataset == "fashion_mnist":
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    else:  # Assume MNIST
        class_names = [str(i) for i in range(10)]
    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix on Test Data")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
