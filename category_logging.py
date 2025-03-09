import numpy as np
import wandb
from tensorflow.keras.datasets import fashion_mnist, mnist
from arguments import get_args

def cat_logging():
    # Get command line arguments
    args = get_args()
    dataset = args.dataset

    # Load dataset based on provided argument
    if dataset == "fashion_mnist":
        (X, y), (_, _) = fashion_mnist.load_data()
    elif dataset == "mnist":
        (X, y), (_, _) = mnist.load_data()
    else:
        raise ValueError("Dataset must be either 'mnist' or 'fashion_mnist'.")

   
    # Get unique labels from the dataset
    unique_labels = np.unique(y)
    images_to_log = []

    # Loop through each unique category label
    for label in unique_labels:
        # Find the index of the first occurrence of this label
        idx = np.where(y == label)[0][0]
        image = X[idx]
        
        # Prepare caption based on dataset type
        if dataset == "mnist":
            caption = f"MNIST Digit: {label}"
        else:
            caption = f"Fashion MNIST Category: {label}"
        
        # Append the image with its caption to the list
        images_to_log.append(wandb.Image(image, caption=caption))
    
    # Log the grid of images to Weights & Biases under a single key
    wandb.log({"Category Log": images_to_log})

