from feedforward_neural_network import FeedforwardNeuralNetwork
from dataset_loader import load_data
from arguments import get_args
import numpy as np
import wandb
from sweep_config import sweep_config  # Import the sweep config from a separate file
from train_sweep import train_sweep  # Import the training function

args=get_args()
if args.perform_sweep:
    # Initialize the W&B sweep (Only run this once per sweep)
    sweep_id = wandb.sweep(sweep_config, project="Fashion-MNIST-Sweep")
    print(f"Initialized sweep with ID: {sweep_id}")

    # Run multiple experiments using W&B Agent
    wandb.agent(sweep_id, function=train_sweep, count=50) 


dataset=args.dataset

X_train, Y_train,X_test, Y_test = load_data(dataset)

# Best parameters from sweep
layer_sizes = [784,128, 128, 128,10]         # Network architecture: input layer, one hidden layer, output layer
activation = "relu"                  # Activation function for hidden layers
weight_init = "random"               # Weight initialization method: "random", "xavier", etc.
learning_rate = 0.001                # Learning rate
weight_decay = 0.5                     # Weight decay 
batch_size = 32                      # Batch size
optimizer = "nadam"                   # Optimizer: "sgd", "momentum", "nag", "rmsprop", "adam", "nadam"
epochs = 10                          # Number of training epochs
args=get_args()
dataset = args.dataset
# Initialize the network using your FeedforwardNeuralNetwork class.
nn = FeedforwardNeuralNetwork(layer_sizes, activation, weight_init, learning_rate, weight_decay, batch_size, optimizer)
nn.train(X_train,Y_train,epochs,batch_size) 

y_pred_train = nn.predict(X_train)
y_true_train = np.argmax(Y_train, axis=1)


y_pred = nn.predict(X_test)
y_true = np.argmax(Y_test, axis=1)
    
args = get_args()

# Set class names based on the dataset
if args.dataset == "fashion_mnist":
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
else:  # Assume MNIST
    class_names = [str(i) for i in range(10)]




wandb.init(project="Fashion-MNIST-Sweep", entity="da24s019-indian-institute-of-technology-madras")

wandb.log({
    "conf_mat_train_data": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_true_train,
        preds=y_pred_train,
        class_names=class_names
    )
})
wandb.log({"conf_mat_test_data" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_true, preds=y_pred,
                        class_names=class_names)})

wandb.finish()