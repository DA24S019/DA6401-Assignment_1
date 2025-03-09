import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network with W&B")

    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname",
                        help="Project name for W&B")s
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname",
                        help="W&B Entity")

    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",
                        help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",
                        help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str,
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="sgd",
                        help="Optimizer")

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for RMSprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for Adam/Nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for Adam/Nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for optimizers")

    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.9, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random",
                        help="Weight initialization")

    parser.add_argument("-nhl", "--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=4, help="Hidden layer size")
    parser.add_argument("-a", "--activation", type=str,
                        choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid",
                        help="Activation function")
    parser.add_argument("-ps","--perform_sweep", action="store_true",  help="If specified, perform hyperparameter sweep for the network")
    args, unknown = parser.parse_known_args()  
    return args
