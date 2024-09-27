import os
import random
import string

import numpy as np
import wandb


def save_all_to_wandb(model, epoch):
    try:
        # Get the needed weights from the model
        needed_weights = model.get_needed_weights()

        # Iterate over the layers' weights
        for layer_name, weights in needed_weights.items():
            # Convert the weights to a NumPy array after detaching from the computation graph
            weights_np = weights.detach().cpu().numpy()

            # Save the weights as a NumPy file
            random_file_name = generate_random_filename()
            weight_file = f"{layer_name}_weights_epoch_{epoch}_{random_file_name}.npy"
            np.save(weight_file, weights_np)

            # Create a wandb artifact for each layer
            artifact = wandb.Artifact(f"{layer_name}_weights_epoch_{epoch}", type="weights")
            artifact.add_file(weight_file)

            # Log the artifact
            wandb.log_artifact(artifact)

            os.remove(weight_file)
    except Exception as e:
        print(e)

    print(f"Saved all needed weights for epoch {epoch}")


def generate_random_filename(length=6):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return f"{random_string}"

import matplotlib.pyplot as plt

def display_weights_from_npy(file_path):

    weights = np.load(file_path)
    weights = (1 / np.max(weights)) * weights
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

#display_weights_from_npy('../y4i_weights_epoch_8_xONPC9.npy')