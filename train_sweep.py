import wandb
from feedforward_neural_network import FeedforwardNeuralNetwork
from dataset_loader import load_data  
from arguments import get_args
from sklearn.model_selection import train_test_split

def train_sweep():
    with wandb.init() as run:
        config = wandb.config

        # Set a meaningful run name
        wandb.run.name = (
            f"ep_{config.epochs}_hl_{config.num_hidden_layers}_hs_{config.hidden_layer_size}_"
            f"wd_{config.weight_decay}_lr_{config.learning_rate}_opt_{config.optimizer}_"
            f"bs_{config.batch_size}_wi_{config.weight_init}_act_{config.activation}"
        )
        args=get_args()
        dataset=args.dataset
        # Load dataset
        X_train, y_train_onehot, _, _ = load_data(dataset)

        X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
        X_train, y_train_onehot, test_size=0.2, random_state=42
    )

        # Initialize the Neural Network
        nn = FeedforwardNeuralNetwork(
            layer_sizes=[784] + [config.hidden_layer_size] * config.num_hidden_layers + [10],
            activation=config.activation,
            weight_decay=config.weight_decay,
            init_method=config.weight_init,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            optimizer=config.optimizer
        )

        # Training loop
        for epoch in range(config.epochs):
            loss, accuracy = nn.train(X_train, y_train_onehot, epochs=1, batch_size=config.batch_size)
            val_loss,val_accuracy = nn.evaluate(X_val, y_val_onehot)

            # Log metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": loss,
                "train_accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
