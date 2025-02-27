import wandb

# Define sweep configuration
sweep_config = {
    "method": "random",  # Can be "grid", "random", or "bayes"
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "epochs": {"values": [5, 10, 15]},
        "hidden_layers": {"values": [2, 3, 4]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="Fashion-MNIST-Sweep")
print(f"Initialized sweep with ID: {sweep_id}")
