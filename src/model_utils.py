"""Model definitions and training helpers (PyTorch MLP + CNN, scikit-learn SVM).

The architecture classes live here (not in the notebook) so the app can rebuild
exactly the same network before loading the saved weights.
"""
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score

from data_preprocessing import IMG_SIZE, N_CLASSES

DEVICE = torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """Fully connected layers only: 784 -> hidden... -> 10."""

    def __init__(self, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        layers, in_features = [], IMG_SIZE * IMG_SIZE
        for h in hidden_sizes:
            layers += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout)]
            in_features = h
        layers.append(nn.Linear(in_features, N_CLASSES))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- CNN
class SimpleCNN(nn.Module):
    """Introductory CNN: [Conv-ReLU-MaxPool] x2 -> Flatten -> FC -> Dropout -> FC."""

    def __init__(self, filters=(16, 32), fc_units=64, dropout=0.3):
        super().__init__()
        f1, f2 = filters
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(f1, f2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f2 * 7 * 7, fc_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_units, N_CLASSES),  # logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- training
def _batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i:i + batch_size]
        yield x[b], y[b]


@torch.no_grad()
def predict_proba(model, x, batch_size=2048):
    model.eval()
    out = [torch.softmax(model(xb), dim=1) for xb, _ in _batches(x, torch.zeros(len(x)), batch_size, False)]
    return torch.cat(out).numpy()


def train_torch(model, x_train, y_train, x_val, y_val, *, lr, batch_size, epochs,
                weight_decay=0.0, optimizer="adam", seed=42, verbose=False, restore_best=False):
    """Train a torch model; return (model, history dict). Inputs are numpy arrays.

    history holds per-epoch train loss / train acc / val acc and the wall-clock training time.
    With restore_best=True the weights of the epoch with the best validation accuracy are
    restored at the end (a form of early stopping; history["best_epoch"] is 1-based).
    """
    set_seed(seed)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train).long()
    xv = torch.from_numpy(x_val)
    model = model.to(DEVICE)
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_acc": []}
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in _batches(xt, yt, batch_size, True):
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        hist["train_loss"].append(total_loss / len(xt))
        hist["train_acc"].append(correct / len(xt))
        hist["val_acc"].append(accuracy_score(y_val, predict_proba(model, xv).argmax(1)))
        if restore_best and hist["val_acc"][-1] >= max(hist["val_acc"]):
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch + 1:2d}  loss {hist['train_loss'][-1]:.4f}  "
                  f"train acc {hist['train_acc'][-1]:.4f}  val acc {hist['val_acc'][-1]:.4f}")
    hist["train_time_s"] = time.perf_counter() - start
    hist["best_epoch"] = int(np.argmax(hist["val_acc"])) + 1
    if restore_best:
        model.load_state_dict(best_state)
    return model, hist
