import numpy as np
from tensorflow.keras.datasets import fashion_mnist, mnist
from sklearn.model_selection import train_test_split

def load_data(dataset_name="fashion_mnist"):
    """
    Loads and preprocesses the dataset.

    Args:
        dataset_name (str): Either "fashion_mnist" or "mnist"

    Returns:
        X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot
    """

    # Load dataset
    if dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError("Invalid dataset. Choose from ['fashion_mnist', 'mnist'].")

    # Normalize pixel values (0-255) → (0-1)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Flatten 28x28 images to 784-dimensional vectors
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)

    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    # Split train data into training (90%) and validation (10%) sets
    X_train, X_val, y_train_onehot, y_val_onehot = train_test_split(
        X_train, y_train_onehot, test_size=0.1, random_state=42
    )

    return X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot
