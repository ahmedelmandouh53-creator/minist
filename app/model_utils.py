"""MLP and CNN definitions. They must match the ones in notebooks 02 and 03, because the app rebuilds
the same network before loading the saved weights.
"""
from torch import nn

from image_utils import IMG_SIZE, N_CLASSES


class MLP(nn.Module):
    """Fully connected layers only: 784 -> hidden... -> 10."""

    def __init__(self, hidden_sizes=(256, 128), dropout=0.2):
        super().__init__()
        layers, in_features = [], IMG_SIZE * IMG_SIZE
        for h in hidden_sizes:
            layers += [nn.Linear(in_features, h), nn.ReLU(), nn.Dropout(dropout)]
            in_features = h
        layers.append(nn.Linear(in_features, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Introductory CNN: [Conv-ReLU-MaxPool] x2 -> Flatten -> FC -> Dropout -> FC."""

    def __init__(self, filters=(16, 32), fc_units=64, dropout=0.3):
        super().__init__()
        f1, f2 = filters
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(f1, f2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f2 * 7 * 7, fc_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_units, N_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
