"""MNIST loading and preprocessing helpers shared by the notebook and the app.

Preprocessing contract (used for training AND inference):
    * pixels are float32 in [0, 1]  (raw uint8 value / 255)
    * digit is WHITE (1.0) on a BLACK (0.0) background, like MNIST itself
    * image is 28 x 28; the MLP / SVM receive it flattened to 784 values,
      the CNN receives it as (1, 28, 28)  (channel, height, width)
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

IMG_SIZE = 28
N_CLASSES = 10
PIXEL_MAX = 255.0
SEED = 42


def load_raw_mnist(data_dir=DATA_DIR):
    """Download (first run only) and return raw uint8 arrays.

    Returns x_train (60000,28,28), y_train (60000,), x_test (10000,28,28), y_test (10000,).
    """
    from torchvision import datasets  # imported here so the app does not need torchvision

    train = datasets.MNIST(str(data_dir), train=True, download=True)
    test = datasets.MNIST(str(data_dir), train=False, download=True)
    return (train.data.numpy(), train.targets.numpy(),
            test.data.numpy(), test.targets.numpy())


def scale_pixels(images_uint8):
    """uint8 [0,255] -> float32 [0,1]."""
    return images_uint8.astype(np.float32) / PIXEL_MAX


def flatten(images):
    """(N,28,28) -> (N,784) for the SVM and the MLP."""
    return images.reshape(len(images), -1)


def add_channel(images):
    """(N,28,28) -> (N,1,28,28) for the CNN (PyTorch is channels-first)."""
    return images[:, None, :, :]


def train_val_split(x, y, val_size=6000, seed=SEED):
    """Stratified split of the 60k training set into train / validation.

    The 10k official test set is never touched here; it is only used for the final evaluation.
    """
    return train_test_split(x, y, test_size=val_size, stratify=y, random_state=seed)
