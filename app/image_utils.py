"""Image constants and reshaping helpers used by the app.

Preprocessing contract (same as in the training notebooks):
    * pixels are float32 in [0, 1]  (raw uint8 value / 255)
    * digit is WHITE (1.0) on a BLACK (0.0) background, like MNIST itself
    * image is 28 x 28; the MLP / SVM receive it flattened to 784 values,
      the CNN receives it as (1, 28, 28)  (channel, height, width)
"""
IMG_SIZE = 28
N_CLASSES = 10
PIXEL_MAX = 255.0


def flatten(images):
    """(N,28,28) -> (N,784) for the SVM and the MLP."""
    return images.reshape(len(images), -1)


def add_channel(images):
    """(N,28,28) -> (N,1,28,28) for the CNN (PyTorch is channels-first)."""
    return images[:, None, :, :]
