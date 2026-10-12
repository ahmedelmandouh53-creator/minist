"""Inference: load the three saved models and turn a drawn canvas into a prediction.

Used by the web app and by the smoke tests at the end of the notebook.
"""
import base64
import io
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from data_preprocessing import IMG_SIZE, PIXEL_MAX, add_channel, flatten
from model_utils import MLP, SimpleCNN

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILES = {
    "svm": MODELS_DIR / "classical_svm.joblib",
    "mlp": MODELS_DIR / "mlp_model.pt",
    "cnn": MODELS_DIR / "cnn_model.pt",
}
MODEL_LABELS = {
    "svm": "Classical ML - SVM (RBF kernel)",
    "mlp": "Neural network - MLP",
    "cnn": "Deep learning - CNN",
}

DIGIT_BOX = 20          # MNIST digits are size-normalised to fit a 20x20 box ...
INK_THRESHOLD = 10      # pixel values <= this are treated as empty background


def canvas_to_mnist(img):
    """Convert a canvas image to a 28x28 float32 array in [0,1] formatted like MNIST.

    MNIST digits are cropped to their bounding box, scaled to fit a 20x20 box (aspect ratio
    kept) and then placed in a 28x28 image so that the centre of mass is at the image centre.
    The canvas is white-on-black (the app draws white strokes on a black canvas), so no colour
    inversion is needed. Returns None if the canvas is empty.
    """
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    ys, xs = np.where(gray > INK_THRESHOLD)
    if len(ys) == 0:
        return None
    digit = Image.fromarray(gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    w, h = digit.size
    scale = DIGIT_BOX / max(w, h)
    digit = digit.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dh, dw = digit.size[1], digit.size[0]
    top, left = (IMG_SIZE - dh) // 2, (IMG_SIZE - dw) // 2
    canvas[top:top + dh, left:left + dw] = np.asarray(digit, dtype=np.float32)

    # shift so the centre of mass is the image centre (as in the original MNIST preparation)
    total = canvas.sum()
    cy, cx = (np.arange(IMG_SIZE) @ canvas.sum(1)) / total, (np.arange(IMG_SIZE) @ canvas.sum(0)) / total
    shift_y, shift_x = int(round(IMG_SIZE / 2 - 0.5 - cy)), int(round(IMG_SIZE / 2 - 0.5 - cx))
    canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
    return np.clip(canvas / PIXEL_MAX, 0.0, 1.0)


def data_url_to_image(data_url):
    """'data:image/png;base64,....' -> PIL image (transparent pixels become black)."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))
    return img


def _load_torch_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cls = {"mlp": MLP, "cnn": SimpleCNN}[ckpt["arch"]]
    model = cls(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


class DigitRecognizer:
    """Holds all three loaded models and predicts from a 28x28 [0,1] image."""

    def __init__(self):
        self.models = {
            "svm": joblib.load(MODEL_FILES["svm"]),
            "mlp": _load_torch_model(MODEL_FILES["mlp"]),
            "cnn": _load_torch_model(MODEL_FILES["cnn"]),
        }

    def predict(self, key, image28):
        """Return (digit, probabilities[10]) from the model named `key`."""
        x = image28[None].astype(np.float32)                       # (1,28,28)
        if key == "svm":
            # SVC was trained with probability=False; softmax of decision scores gives a confidence
            scores = self.models["svm"].decision_function(flatten(x))[0]
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
        else:
            with torch.no_grad():
                inp = torch.from_numpy(flatten(x) if key == "mlp" else add_channel(x))
                probs = torch.softmax(self.models[key](inp), dim=1)[0].numpy()
        return int(np.argmax(probs)), probs
