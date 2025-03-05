import wandb
from feedforward_neural_network import FeedforwardNeuralNetwork
from dataset_loader import load_data  # Load dataset dynamically

def train_sweep():
    with wandb.init() as run:
        config = wandb.config

        # Set a meaningful run name
        wandb.run.name = (
            f"ep_{config.epochs}_hl_{config.num_hidden_layers}_hs_{config.hidden_layer_size}_"
            f"wd_{config.weight_decay}_lr_{config.learning_rate}_opt_{config.optimizer}_"
            f"bs_{config.batch_size}_wi_{config.weight_init}_act_{config.activation}"
        )

        # Load dataset
        X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot = load_data("fashion_mnist")

        # Initialize the Neural Network
        nn = FeedforwardNeuralNetwork(
            layer_sizes=[784] + [config.hidden_layer_size] * config.num_hidden_layers + [10],
            activation=config.activation,
            init_method=config.weight_init,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer
        )

        # Training loop
        for epoch in range(config.epochs):
            loss, accuracy = nn.train(X_train, y_train_onehot, epochs=1, batch_size=config.batch_size)
            val_loss = nn.compute_loss(X_val, y_val_onehot)
            val_accuracy = nn.compute_accuracy(X_val, y_val_onehot)

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
